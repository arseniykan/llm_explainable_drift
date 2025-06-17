# Explainable Concept Drift in Process Mining

Welcome to the repository for "LLM-Augmented Concept Drift Detection for Object Centric Process Mining"! This project extends existing frameworks for concept drift detection in process mining by integrating Large Language Models (LLMs) to provide human-readable, causal explanations and actionable insights.

Building upon the work by Adams et al. (2023), which uses Granger causality and the PELT algorithm on object-centric logs, our method enhances interpretability. For each statistically explainable drift, we leverage LLMs to generate detailed explanations, focusing on both quantitative and qualitative aspects, and investigating the impact of providing rich contextual information.

## Project Overview

This repository provides the code to reproduce the experiments, results, and figures presented in the corresponding research paper. Our key contributions include:

*   **LLM-Augmented Explanations**: Generating natural language explanations for detected concept drifts, making complex statistical findings accessible to domain experts.
*   **Targeted Prompt Design**: A systematic approach to crafting specific LLM prompts to elicit both quantitative (e.g., percentage changes) and qualitative (e.g., business impact, causal relationships, recommendations) responses.
*   **Contextual Impact Analysis**: Investigating how providing varying levels of domain-specific context within prompts influences the quality and relevance of LLM-generated explanations.
*   **Multi-LLM Support**: Integration with OpenAI (GPT), Anthropic (Claude), and Google (Gemini) models for comparative analysis of their explanatory capabilities.

## Code Structure

*   `llm_explainer.py`: The core Python script responsible for loading drift data, constructing prompts, calling various LLM APIs, and saving the generated explanations.
*   `prompts.json`: A JSON file containing the templates for different prompt categories (quantitative, qualitative) and types (plain data, context-rich). This allows for flexible and systematic prompt management.
*   `drift_results.json`: An example input file containing pre-detected concept drift data, including time series values and Granger causality p-values.
*   `llm_explanations.json`: The output file where all LLM-generated explanations are saved in JSON format.
*   `environment.yml`: Conda environment configuration file for easy dependency management.
*   `experiments.py`: (Assumed from user input) Script to reproduce the main experiments and generate figures related to drift detection.

## Quickstart

Follow these steps to set up the environment and run the experiments:

### 1. Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### 2. Data Preparation

First, unzip `example_logs/mdl/BPI2017.zip` into the same directory, i.e., `example_logs/mdl/BPI2017.csv`.

### 3. Environment Setup

Create and activate the Conda environment by running the following commands in your terminal:

```bash
conda env create --file environment.yml
conda activate explainable_concept_drift_experiments
```

### 4. API Keys Configuration

This project uses external LLM APIs. You need to set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export GOOGLE_API_KEY="your_google_api_key"
```

Replace `"your_openai_api_key"`, `"your_anthropic_api_key"`, and `"your_google_api_key"` with your actual API keys.

### 5. Run Experiments

Navigate to the root directory of this repository and run the main experiment script:

```bash
python experiments.py
```

This will reproduce the core drift detection and analysis. To generate LLM explanations based on the detected drifts, run:

```bash
python llm_explainer.py
```

This script will generate `llm_explanations.json` containing the LLM outputs for various prompt strategies and models.

## Contact

For any questions or inquiries, please contact [arseniykan@unist.ac.kr](mailto:arseniykan@unist.ac.kr).


