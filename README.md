# LiveProteinBench

LiveProteinBench is a benchmark suite designed for evaluating Large Language Models (LLMs) on protein-related tasks. This project aims to assess the capabilities of models in understanding protein sequences, structural images, and predicting various biological functions.

## Introduction

This repository provides an automated evaluation pipeline that leverages OpenAI-compatible APIs to reason about and answer specific protein biology questions.

It supports two evaluation modes through one unified runner:
1.  **Multimodal Evaluation (`chat.py --use-images`)**: Analyzes both protein sequences and protein structure images.
2.  **Text-only Evaluation (`chat.py --no-images`)**: Analyzes protein sequences only.

## Supported Tasks

The benchmark covers a wide range of biological prediction tasks (defined in `prompt.json`), including:

-   **Cofactor Prediction**: Identifying specific cofactors required for enzyme function.
-   **Transmembrane Region Prediction**: Predicting protein transmembrane domains.
-   **Active Site Prediction**: Identifying key amino acid residues in the enzyme active site.
-   **EC Number Prediction**: Predicting the Enzyme Commission (EC) number.
-   **Catalytic Activity**: Identifying the specific chemical reaction catalyzed by an enzyme.
-   **Motif Position Prediction**: Identifying and locating conserved sequence motifs.
-   **Pathway Prediction**: Predicting the metabolic or signaling pathway of a protein.
-   **pH Dependence**: Predicting the optimal pH for protein activity.
-   **Temperature Dependence**: Predicting the thermal adaptation category of a protein.
-   **Gene Ontology (GO) Prediction**: Including Molecular Function (MF), Cellular Component (CC), and Biological Process (BP).

## Repository Structure

```
LiveProteinBench/
├── chat.py                   # Unified evaluation script for multimodal/text-only runs
├── run_chat.sh               # Example shell launcher for the local chat.py command
├── prompt.json               # Task definitions and prompt templates
└── dataset/                  # Original dataset files
    ├── generate_images.py    # Generate structure images for the dataset
    └── QA/                   # Question data for each task

```

## Usage

Before running the scripts, make sure you have your input data prepared in the `dataset/QA/` directory and have configured your API credentials. For multimodal runs, execute from the repository root so the relative image paths inside the dataset resolve correctly.

### API Configuration

You can pass the API configuration through command-line arguments or environment variables:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="YOUR_API_BASE_URL"
```

You can then choose the target model(s) and task(s) at runtime. The default LLM sampling temperature is `0.2`, and you can override it with `--temperature`.

### Running Multimodal Evaluation

This mode reads protein sequences and encodes corresponding structure images (Base64) to query vision-capable models.

```bash
python chat.py --models claude-3-7-sonnet-20250219 --tasks EC_number cofactor --use-images --temperature 0.2
```

### Running Text-only Evaluation

This mode uses only the protein sequence text as input, suitable for text-only models or sequence-based analysis.

```bash
python chat.py --models deepseek-v3 --tasks Enzyme_function EC_number --no-images --temperature 0.2
```

### Shell Launcher

The repository also includes a simple shell launcher that mirrors a fixed local command for `chat.py`. Edit `run_chat.sh` if you want to change the default tasks, model, or output path.

```bash
bash run_chat.sh
```

Current contents of `run_chat.sh`:

```bash
python chat.py \
  --tasks pathway temperature \
  --models claude-3-7-sonnet-20250219 \
  --data-dir ./dataset/QA \
  --prompt-file ./prompt.json \
  --temperature 0.2 \
  --output-root ./responses
```

## Output

Evaluation results are saved in the output directory specified by `--output-root` (default: `responses/`). The results are organized by mode, task name, and model name, containing the detailed reasoning, final answer, token usage, and the temperature used for each protein.

*   **Individual result**: `responses/<mode>/<task_name>/<model_name>/<protein_id>.json`
*   **Aggregated result**: `responses/<mode>/<task_name>/<model_name>/<task_name>_all_questions.json`
