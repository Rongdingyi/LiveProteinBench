# LiveProteinBench

LiveProteinBench is a benchmark suite designed for evaluating Large Language Models (LLMs) on protein-related tasks. This project aims to assess the capabilities of models in understanding protein sequences, structural images, and predicting various biological functions.

## Introduction

This repository provides an automated evaluation pipeline that leverages OpenAI-compatible APIs (such as GPT-4, Claude 3.5 Sonnet, DeepSeek, etc.) to reason about and answer specific protein biology questions.

It supports two main evaluation modes:
1.  **Multimodal Evaluation (`chat.py`)**: Analyzes both protein sequences and protein structure images.
2.  **Text-only Evaluation (`chat_nopic.py`)**: Analyzes protein sequences only.

## Supported Tasks

The benchmark covers a wide range of biological prediction tasks (defined in `prompt.json`), including:

-   **Cofactor Prediction**: Identifying specific cofactors required for enzyme function.
-   **Transmembrane Region Prediction**: Predicting protein transmembrane domains.
-   **Active Site Prediction**: Identifying key amino acid residues in the enzyme active site.
-   **EC Number Prediction**: Predicting the Enzyme Commission (EC) number.
-   **Catalytic Activity**: Identifying the specific chemical reaction catalyzed by an enzyme.
-   **Motif Identification & Position**: Identifying and locating conserved sequence motifs.
-   **Pathway Prediction**: Predicting the metabolic or signaling pathway of a protein.
-   **pH Dependence**: Predicting the optimal pH for protein activity.
-   **Temperature Dependence**: Predicting the thermal adaptation category of a protein.
-   **Gene Ontology (GO) Prediction**: Including Molecular Function (MF), Cellular Component (CC), and Biological Process (BP).

## Repository Structure

```
LiveProteinBench/
├── chat.py                   # Multimodal evaluation script (Sequence + Images)
├── chat_nopic.py             # Text-only evaluation script (Sequence only)
├── prompt.json               # Task definitions and prompt templates
└── dataset/                  # Original dataset files
    ├── generate_images.py    # Generate structure images for the dataset
    ├── origin_data/          # CSV data for each task
    └── QA/                   # Question data for each task

```

## Usage

Before running the scripts, make sure you have your input data prepared in the `QA/` directory and have configured your API credentials.

### API Configuration

Open `chat.py` or `chat_nopic.py` and configure the client with your Base URL and API Key:

```python
client = OpenAI(
    base_url="YOUR_API_BASE_URL",
    api_key="YOUR_API_KEY"
)
```

You can also modify the `models` list in the `main` function to specify which models you want to evaluate (e.g., `claude-3-5-sonnet-20240620`, `deepseek-v3`).

### Running Multimodal Evaluation

This script reads protein sequences and encodes corresponding structure images (Base64) to query vision-capable models.

```bash
python chat.py
```

### Running Text-only Evaluation

This script uses only the protein sequence text as input, suitable for text-only models or sequence-based analysis.

```bash
python chat_nopic.py
```

## Output

Evaluation results are saved in the `responses/` directory. The results are organized by task name and model name, containing the detailed reasoning, final answer, and token usage for each protein.

*   **Individual result**: `responses/<task_name>/<model_name>/<protein_id>.json`
*   **Aggregated result**: `responses/<task_name>/<model_name>/<task_name>_all_questions.json`
