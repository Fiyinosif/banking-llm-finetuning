# Banking Customer Support LLM

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

## Project Overview

This project explores parameter-efficient fine-tuning of a Large Language Model (LLM) for retail banking customer support. The objective is to adapt a pretrained instruction-following model to generate accurate, helpful, and context-aware responses to customer banking inquiries.

The project follows a standard machine learning workflow:

1. Evaluate a pretrained model to establish a baseline.
2. Fine-tune the model on a retail banking support dataset.
3. Re-evaluate the fine-tuned model using the same evaluation pipeline.
4. Compare performance against the baseline.

---

## Project Goal

The primary goal of this project is to investigate whether fine-tuning an open-source LLM can improve customer support response quality for retail banking tasks.

Success will be measured by comparing the baseline and fine-tuned models using semantic evaluation metrics, with BERTScore serving as the primary metric.

---

## Dataset

**Dataset:** `sohamb37lexsi/bitext-retail-banking-llm-chatbot-splits`

The dataset contains real-world banking support instructions paired with high-quality reference responses.

### Dataset Split

| Split | Samples |
|-------:|--------:|
| Train | 20,436 |
| Validation | 2,554 |
| Test | 2,555 |

Each example contains:

- `instruction` – Customer request or question
- `response` – Expected customer support response
- `intent` – Banking intent label
- `category` – Banking category
- `tags` – Additional metadata

---

## Base Model

**Model:** `Qwen2.5-0.5B-Instruct`

The pretrained model was evaluated **without any task-specific fine-tuning** to establish a baseline.

For each sample in the test dataset:

- The customer instruction was tokenized.
- A response was generated using the pretrained model.
- The generated response was cleaned by removing the original prompt.
- Predictions and corresponding ground-truth responses were saved to `results/baseline_evaluation.csv`.

---

## Baseline Results

The baseline model was evaluated on all **2,555** test samples using **BERTScore**, which measures semantic similarity between generated and reference responses.

### Baseline Metrics

| Metric | Score |
|--------|------:|
| Precision | **0.8386** |
| Recall | **0.8432** |
| **F1 Score** | **0.8407** |

These results serve as the benchmark for evaluating the effectiveness of fine-tuning.

---

## Fine-Tuning

**Status:** 🚧 In Progress

The next phase of this project will fine-tune the base model using parameter-efficient fine-tuning techniques (LoRA/QLoRA).

The same evaluation pipeline will be reused after training to allow for a direct comparison with the baseline.

---

## Evaluation

Model performance is evaluated using **BERTScore**.

Unlike exact-match metrics, BERTScore compares the semantic meaning of generated and reference responses using contextual embeddings from **RoBERTa-large**.

Because the evaluation dataset contains over 2,500 examples, BERTScore is computed in batches to reduce memory usage while loading the RoBERTa evaluation model only once.

Evaluation outputs are saved as:

- `results/baseline_evaluation.csv`
- `results/baseline_metrics.csv`

---

## Results

| Model | Precision | Recall | BERTScore F1 |
|------|----------:|-------:|-------------:|
| Qwen2.5-0.5B-Instruct (Baseline) | 0.8386 | 0.8432 | **0.8407** |
| Fine-Tuned Model | - | - | - |

---

## Future Improvements

Potential improvements include:

- Fine-tune using LoRA or QLoRA
- Hyperparameter optimization
- Evaluate using additional metrics (ROUGE, BLEU)
- Compare multiple open-source LLMs
- Perform human evaluation of generated responses
- Deploy the fine-tuned model as a banking support chatbot

---

## Why I Chose This Dataset

I selected this dataset because it provides high-quality instruction-response pairs specifically focused on retail banking customer support.

Unlike general conversational datasets, the responses emphasize accurate, professional, and task-oriented assistance. This makes the dataset well suited for studying domain adaptation through LLM fine-tuning while providing a realistic benchmark for evaluating improvements after training.
