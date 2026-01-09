import base64
import glob
import os
from openai import OpenAI
import json

client = OpenAI(
    base_url="XXX",
    api_key="XXX"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_json_files(output_dir):
    if not os.path.exists(output_dir):
        return []
    json_paths = glob.glob(os.path.join(output_dir, "*.json"))
    json_filenames = [os.path.splitext(os.path.basename(path))[0] for path in json_paths]
    return json_filenames

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_questions(model_name, json_data, task_name, task_description, file_name, output_dir):
    results = []
    json_files = get_json_files(output_dir)
    for i, item in enumerate(json_data):
        try:
            protein_name = item["Protein Image"][0].split('/')[-2]
            if protein_name in json_files:
                continue

            protein_sequence = item["Protein Sequence"]   
            protein_images = item["Protein Image"]
            
            choices = item["Multiple Choices"]
            choices_text = "\n".join([f"{key}. {value}" for key, value in choices.items()])
            system_prompt = f"""You are an excellent scientist. Your should analyze the provided protein-related task carefully and choose the correct answer form the multiple-choice options.

                            The current task is about {task_name}, which {task_description}. The inputs provided by the user for this task include:

                            * Protein Sequence: The amino acid sequence of the protein.
                            * Protein Image: Multiple views of the protein structure.
                            * Multiple Choices: Options for the answer.

                            Please think step-by-step about this problem:
                            1. Analyze the protein sequence and structure carefully
                            2. Consider the biological context and function
                            3. Evaluate each multiple choice option
                            4. Provide your reasoning process
                            5. Finally, give your answer

                            Provide your response in following format:
                            1. reasoning: [Your detailed reasoning here]
                            2. answer: Your final answer, which should be \"A\", \"B\", \"C\" or \"D\".
                            """

            image_content = []
            for idx, img_path in enumerate(protein_images[:6]):
                base64_img = encode_image(img_path)
                image_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}"
                    }
                })

            
            choices_text = "\n".join([f"{key}. {value}" for key, value in choices.items()])
            
            task_specific_user_prompt = f"""
                                        [Protein Sequence]
                                        {protein_sequence}

                                        [Protein Image]
                                        (Images are provided as attachments)

                                        [Multiple Choices]
                                        {choices_text}
                                        """
            messages_content = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": task_specific_user_prompt}] + image_content
                }
            ]
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages_content,
                    max_tokens=10240,
                )
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens_this_request = usage.total_tokens
                content = response.choices[0].message.content
                response_dict = {
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content
                            },
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    }
                }
                detailed_result = {
                    "response": content,
                    "output": response_dict,
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens_this_request,
                    },
                    "input": messages_content
                }
                output_path = os.path.join(output_dir, f"{protein_name}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_result, f, indent=2, ensure_ascii=False)
                results.append({
                    "system_prompt": system_prompt,
                    "user_prompt": task_specific_user_prompt,
                    "choices": choices,
                    "response": content,
                    "protein_id": protein_name
                })
            
            except Exception as api_final_err:
                with open(os.path.join(output_dir, "failed_proteins.txt"), "a") as f:
                    f.write(f"{protein_name}: {str(api_final_err)}\n")
        
        except Exception as item_err:
            continue
    
    return results

def get_json_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        return []
    
    json_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            json_files.append(os.path.splitext(filename)[0])
    
    return json_files

def read_prompt_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return None

def main():
    models = ['claude-3-7-sonnet-20250219']
    prompt_data = read_prompt_json("./prompt.json")
    task_dict = {task['task']: task for task in prompt_data}
    for file_name in task_dict.keys():
        for model_name in models:
            input_file = "./QA/" + file_name +".json"
            output_dir = "./responses/" + file_name+ '/' + model_name
            task_name = task_dict[file_name]['task_name']
            task_description = task_dict[file_name]['task_description']
            json_data = read_json_file(input_file)
            os.makedirs(output_dir, exist_ok=True)
            results = generate_questions(model_name, json_data, task_name, task_description, file_name, output_dir)

            all_output_path = os.path.join(output_dir, f"{file_name}_all_questions.json")
            with open(all_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()