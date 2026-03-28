import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_DATA_DIR = "dataset/QA"
DEFAULT_OUTPUT_ROOT = "responses"
DEFAULT_PROMPT_FILE = "prompt.json"
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_MAX_TOKENS = 10240
DEFAULT_MAX_IMAGES = 6
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_TEMPERATURE = 0.2


def parse_multi_values(values: Optional[Sequence[str]]) -> List[str]:
    parsed_values: List[str] = []
    seen = set()
    for value in values or []:
        for part in value.split(","):
            candidate = part.strip()
            if candidate and candidate not in seen:
                parsed_values.append(candidate)
                seen.add(candidate)
    return parsed_values


def sanitize_path_component(value: str) -> str:
    safe_value = value.replace("/", "__").replace("\\", "__").replace(":", "_")
    safe_value = safe_value.replace(" ", "_")
    return safe_value or "unknown"


def read_json_file(file_path: Path):
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(file_path: Path, payload) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def append_failure(output_dir: Path, protein_id: str, error: Exception) -> None:
    failed_path = output_dir / "failed_proteins.txt"
    with failed_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{protein_id}: {error}\n")


def discover_available_tasks(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(path.stem for path in data_dir.glob("*.json"))


def load_task_metadata(prompt_file: Path) -> Dict[str, Dict[str, str]]:
    if not prompt_file.exists():
        return {}

    prompt_data = read_json_file(prompt_file)
    task_metadata = {}
    for task in prompt_data:
        task_key = task.get("task")
        if not task_key:
            continue
        task_metadata[task_key] = {
            "task_name": task.get("task_name", task_key.replace("_", " ").title()),
            "task_description": task.get(
                "task_description",
                "is a protein-related multiple-choice prediction task.",
            ),
        }
    return task_metadata


def resolve_requested_tasks(requested_tasks: List[str], available_tasks: List[str]) -> List[str]:
    if not requested_tasks or "all" in requested_tasks:
        return available_tasks

    missing_tasks = [task for task in requested_tasks if task not in available_tasks]
    if missing_tasks:
        available_text = ", ".join(available_tasks)
        missing_text = ", ".join(missing_tasks)
        raise ValueError(
            f"Task file(s) not found: {missing_text}. Available tasks: {available_text}"
        )
    return requested_tasks


def get_task_info(
    task_name: str,
    task_metadata: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    metadata = task_metadata.get(task_name)
    if metadata:
        return metadata["task_name"], metadata["task_description"]

    fallback_title = task_name.replace("_", " ").title()
    fallback_description = (
        "is a protein-related multiple-choice prediction task. Analyze the "
        "protein inputs carefully and select the best option."
    )
    return fallback_title, fallback_description


def extract_protein_id(item: dict, index: int) -> str:
    protein_images = item.get("Protein Image") or []
    if protein_images:
        first_image = Path(protein_images[0])
        parent_name = first_image.parent.name
        if parent_name:
            return parent_name
        return first_image.stem

    for key in ("Protein ID", "protein_id", "Entry", "entry"):
        if key in item and item[key]:
            return str(item[key])

    return f"sample_{index:05d}"


def build_system_prompt(task_name: str, task_description: str, use_images: bool) -> str:
    input_lines = [
        "* Protein Sequence: The amino acid sequence of the protein.",
        "* Multiple Choices: Candidate answers for the protein-related question.",
    ]
    if use_images:
        input_lines.insert(
            1, "* Protein Image: Multiple views of the protein structure."
        )

    step_one = "Analyze the protein sequence"
    if use_images:
        step_one += " and structure"

    input_description = "\n".join(input_lines)
    return f"""You are an excellent scientist. Analyze the provided protein-related task carefully and choose the correct answer from the multiple-choice options.

The current task is about {task_name}, which {task_description}. The inputs provided by the user for this task include:

{input_description}

Please think step-by-step about this problem:
1. {step_one} carefully.
2. Consider the biological context and function.
3. Evaluate each multiple-choice option.
4. Provide your reasoning process.
5. Finally, give your answer.

Provide your response in the following format:
1. reasoning: [Your detailed reasoning here]
2. answer: Your final answer, which should be "A", "B", "C" or "D".
"""


def build_user_message(
    item: dict,
    use_images: bool,
    max_images: int,
) -> Tuple[List[dict], str, dict]:
    protein_sequence = item["Protein Sequence"]
    choices = item["Multiple Choices"]
    choices_text = "\n".join(f"{key}. {value}" for key, value in choices.items())

    prompt_sections = [
        "[Protein Sequence]",
        protein_sequence,
    ]
    if use_images:
        prompt_sections.extend(["", "[Protein Image]", "(Images are provided as attachments)"])
    prompt_sections.extend(["", "[Multiple Choices]", choices_text])
    user_prompt = "\n".join(prompt_sections)

    content = [{"type": "text", "text": user_prompt}]
    if use_images:
        protein_images = item.get("Protein Image") or []
        for image_path in protein_images[:max_images]:
            base64_image = encode_image(Path(image_path))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

    return content, user_prompt, choices


def create_client(base_url: Optional[str], api_key: Optional[str]) -> Any:
    from openai import OpenAI

    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

    if not resolved_api_key:
        raise ValueError(
            "Missing API key. Please provide --api-key or set OPENAI_API_KEY/API_KEY."
        )

    client_kwargs = {"api_key": resolved_api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    return OpenAI(**client_kwargs)


def call_chat_completion(
    client: Any,
    model_name: str,
    messages_content: List[dict],
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_delay: float,
    reasoning_effort: Optional[str],
):
    request_kwargs = {
        "model": model_name,
        "messages": messages_content,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as error:
            last_error = error
            if attempt == max_retries:
                break
            time.sleep(retry_delay)

    raise last_error


def serialize_response(response) -> dict:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    return {
        "id": response.id,
        "object": response.object,
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                },
                "finish_reason": choice.finish_reason,
            }
            for choice in response.choices
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def generate_questions(
    client: Any,
    model_name: str,
    json_data: List[dict],
    task_key: str,
    task_name: str,
    task_description: str,
    output_dir: Path,
    use_images: bool,
    max_images: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_delay: float,
    reasoning_effort: Optional[str],
) -> Dict[str, int]:
    stats = {"generated": 0, "skipped": 0, "failed": 0}
    safe_model_name = sanitize_path_component(model_name)

    for index, item in enumerate(json_data):
        protein_id = extract_protein_id(item, index)
        safe_protein_id = sanitize_path_component(protein_id)
        output_path = output_dir / f"{safe_protein_id}.json"

        if output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            system_prompt = build_system_prompt(task_name, task_description, use_images)
            user_content, user_prompt, choices = build_user_message(
                item=item,
                use_images=use_images,
                max_images=max_images,
            )
            messages_content = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            response = call_chat_completion(
                client=client,
                model_name=model_name,
                messages_content=messages_content,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
                retry_delay=retry_delay,
                reasoning_effort=reasoning_effort,
            )
            response_content = response.choices[0].message.content
            response_dict = serialize_response(response)
            token_usage = response_dict["usage"]

            detailed_result = {
                "task": task_key,
                "task_name": task_name,
                "task_description": task_description,
                "protein_id": protein_id,
                "model": model_name,
                "model_dir_name": safe_model_name,
                "use_images": use_images,
                "temperature": temperature,
                "response": response_content,
                "summary": {
                    "protein_id": protein_id,
                    "model": model_name,
                    "task": task_key,
                    "task_name": task_name,
                    "use_images": use_images,
                    "temperature": temperature,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "choices": choices,
                    "response": response_content,
                },
                "output": response_dict,
                "token_usage": token_usage,
                "input": messages_content,
            }
            write_json_file(output_path, detailed_result)
            stats["generated"] += 1

        except Exception as error:
            append_failure(output_dir, protein_id, error)
            stats["failed"] += 1

    return stats


def build_aggregate_results(output_dir: Path) -> List[dict]:
    aggregate_results = []
    for json_path in sorted(output_dir.glob("*.json")):
        if json_path.name.endswith("_all_questions.json"):
            continue

        try:
            payload = read_json_file(json_path)
        except Exception:
            continue

        if "summary" in payload:
            aggregate_results.append(payload["summary"])
            continue

        aggregate_results.append(
            {
                "protein_id": payload.get("protein_id", json_path.stem),
                "model": payload.get("model"),
                "task": payload.get("task"),
                "task_name": payload.get("task_name"),
                "use_images": payload.get("use_images"),
                "temperature": payload.get("temperature"),
                "response": payload.get("response"),
            }
        )

    return aggregate_results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LiveProteinBench with one unified script for multimodal and text-only evaluation."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="One or more model names. Supports repeated values or comma-separated input.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["all"],
        help="One or more task names. Supports repeated values, comma-separated input, or 'all'.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing task JSON files.")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="Path to prompt metadata JSON.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for model responses.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum completion tokens.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature for the LLM.")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES, help="Maximum number of protein images to send.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Maximum retries per request.")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY, help="Delay between retries in seconds.")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional reasoning_effort value for supported models.",
    )
    parser.add_argument(
        "--mode-name",
        default=None,
        help="Optional output subdirectory name. Defaults to 'multimodal' or 'text_only'.",
    )

    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--use-images",
        dest="use_images",
        action="store_true",
        help="Run in multimodal mode with protein structure images.",
    )
    image_group.add_argument(
        "--no-images",
        dest="use_images",
        action="store_false",
        help="Run in text-only mode without protein structure images.",
    )
    parser.set_defaults(use_images=True)

    args = parser.parse_args(argv)
    args.models = parse_multi_values(args.models)
    args.tasks = parse_multi_values(args.tasks)
    if not args.models:
        args.models = [DEFAULT_MODEL]
    if not args.tasks:
        args.tasks = ["all"]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    prompt_file = Path(args.prompt_file)
    output_root = Path(args.output_root)
    mode_name = args.mode_name or ("multimodal" if args.use_images else "text_only")

    available_tasks = discover_available_tasks(data_dir)
    selected_tasks = resolve_requested_tasks(args.tasks, available_tasks)
    task_metadata = load_task_metadata(prompt_file)
    client = create_client(base_url=args.base_url, api_key=args.api_key)

    for task_key in selected_tasks:
        input_file = data_dir / f"{task_key}.json"
        task_name, task_description = get_task_info(task_key, task_metadata)
        json_data = read_json_file(input_file)

        for model_name in args.models:
            model_dir_name = sanitize_path_component(model_name)
            output_dir = output_root / mode_name / task_key / model_dir_name
            output_dir.mkdir(parents=True, exist_ok=True)

            stats = generate_questions(
                client=client,
                model_name=model_name,
                json_data=json_data,
                task_key=task_key,
                task_name=task_name,
                task_description=task_description,
                output_dir=output_dir,
                use_images=args.use_images,
                max_images=args.max_images,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                reasoning_effort=args.reasoning_effort,
            )

            aggregate_results = build_aggregate_results(output_dir)
            all_output_path = output_dir / f"{task_key}_all_questions.json"
            write_json_file(all_output_path, aggregate_results)

            print(
                f"[{mode_name}] task={task_key} model={model_name} "
                f"generated={stats['generated']} skipped={stats['skipped']} "
                f"failed={stats['failed']} saved_to={output_dir}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
