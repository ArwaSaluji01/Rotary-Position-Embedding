
# Rotary Position Embedding: A Miniature RoFormer Study

A from-scratch educational implementation and empirical investigation of **Rotary Position Embedding (RoPE)** in a compact Transformer language model.

This project explores how different positional encoding strategies affect causal language modeling performance. It implements a standard Transformer with learned positional embeddings, a Transformer using Rotary Position Embedding, and a no-position ablation model.

The implementation is designed to develop practical understanding of Transformer attention, positional representations, model training, and experimental evaluation.

---

## Project Overview

Transformers use self-attention to model relationships between tokens. However, self-attention alone does not inherently encode the sequential order of tokens. Positional information is therefore required to help the model distinguish between different token positions.

The original **RoFormer: Enhanced Transformer with Rotary Position Embedding** paper introduces RoPE, a positional encoding technique that represents absolute positions through rotations while incorporating relative positional information into the attention mechanism.

In this project, RoPE is implemented from scratch and compared with:

1. **Learned Positional Embeddings**
2. **Rotary Position Embedding (RoPE)**
3. **No Positional Encoding**

The models are trained on the WikiText-2 dataset using a compact character-level causal language modeling setup.

### Project Objectives

- Understand the role of positional information in Transformer architectures.
- Implement Rotary Position Embedding using sine and cosine functions.
- Apply RoPE to query and key representations in self-attention.
- Compare learned positional embeddings with RoPE.
- Investigate the effect of removing positional information.
- Evaluate models using cross-entropy loss and perplexity.
- Analyze qualitative text generation from trained models.
- Identify differences between this implementation and the original RoFormer research.

---

## Original Research Paper

**Paper:** RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors:** Jianlin Su et al.

**Paper link:**  
https://arxiv.org/abs/2104.09864

**Official RoFormer implementation:**  
https://github.com/ZhuiyiTechnology/roformer

### Core Idea

RoPE applies position-dependent rotations to query and key representations in self-attention.

The paper describes how this approach encodes absolute position through a rotation matrix while allowing the attention inner product to incorporate relative position information.

The paper also discusses several properties of RoPE, including:

- Relative position dependency in self-attention.
- Long-term decay of inter-token dependency with increasing relative distance.
- Compatibility with linear attention.
- Application to long-text modeling and other NLP tasks.

This project implements a compact version of the core RoPE mechanism for educational experimentation.

---

## Dataset

### WikiText-2

The project uses the raw WikiText-2 dataset from the Hugging Face Datasets library.

**Dataset:**  
https://huggingface.co/datasets/Salesforce/wikitext

**Configuration:** `wikitext-2-raw-v1`

### Dataset Processing

The following preprocessing steps are applied:

- Use the official training and validation splits.
- Strip leading and trailing whitespace from text lines.
- Remove empty lines.
- Construct a character-level vocabulary.
- Encode characters as integer token IDs.
- Create fixed-length input sequences.
- Train the model to predict the next character.

### Task

This project performs **causal character-level language modeling**.

Given a sequence of characters, the model predicts the next character at each position.

The training objective is cross-entropy loss.

This differs from the masked language modeling and other downstream evaluation tasks used in the original RoFormer paper.

---

## Model Architecture

The implementation uses a compact Transformer architecture suitable for experimentation in Google Colab.

| Configuration | Value |
|---|---:|
| Model type | Decoder-style causal Transformer |
| Tokenization | Character-level |
| Dataset | WikiText-2 |
| Context length | 128 |
| Embedding dimension | 128 |
| Number of attention heads | 4 |
| Number of Transformer layers | 3 |
| Dropout | 0.1 |
| Attention type | Causal self-attention |
| Feed-forward hidden dimension | 4 × model dimension |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

### Compared Models

#### 1. Learned Position Model

The baseline model uses token embeddings and trainable positional embeddings.

The positional embeddings are added to the token representations before they are processed by the Transformer blocks.

#### 2. RoPE Model

The RoPE model does not use learned positional embeddings.

Instead, rotary position encoding is applied to the query and key representations inside the attention mechanism.

The value representations are not rotated.

#### 3. No-Position Model

The ablation model uses token embeddings without learned positional embeddings or RoPE.

This model is included to investigate the effect of removing explicit positional information.

---

## Implementation Details

The notebook includes the following components:

- Character-level tokenizer.
- Input-target sequence generation.
- Causal self-attention.
- Causal attention masking.
- Feed-forward networks.
- Pre-layer-normalized Transformer blocks.
- Learned positional embeddings.
- Rotary position encoding.
- Sine and cosine cache generation.
- Query and key rotation.
- Training and validation loss estimation.
- Gradient clipping.
- Best-validation-checkpoint restoration.
- Text generation using temperature and optional top-k sampling.
- Perplexity calculation.
- Positional encoding ablation.

### RoPE Implementation

The implementation follows the central mechanism described in the paper:

1. Project the input representations into queries, keys, and values.
2. Reshape the query and key tensors into attention heads.
3. Construct position-dependent sine and cosine values.
4. Apply the rotary transformation to queries and keys.
5. Compute causal self-attention.
6. Continue with the remaining Transformer block operations.

The rotation is implemented efficiently using paired dimensions rather than constructing a full rotation matrix.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/rotary-position-embedding-study.git
cd rotary-position-embedding-study
```

Replace `<your-username>` with your GitHub username.

### 2. Create a Python Environment

Python 3.10 or a compatible recent Python version is recommended.

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required libraries:

```bash
pip install torch datasets matplotlib numpy
```

Depending on the notebook environment, additional standard dependencies may already be available.

### 4. Run the Notebook

The project is designed for Google Colab.

1. Open the notebook in Google Colab.
2. Select an available runtime.
3. Run the cells sequentially.
4. Download the dataset when prompted by the Hugging Face Datasets library.
5. Train the baseline and RoPE models.
6. Evaluate the models and generate sample text.

The notebook contains implementation explanations, validation checks, training outputs, and evaluation results.

---

## Experimental Results

The models were evaluated using validation cross-entropy loss and perplexity.

### Validation Results

| Model | Validation Loss | Validation Perplexity |
|---|---:|---:|
| Learned Position | 1.8867 | 6.5978 |
| RoPE | **1.6090** | **4.9981** |
| No Position | 2.3836 | 10.8437 |

### Training Configuration

The learned-position and RoPE models were trained for 3,000 steps.

The no-position ablation model was trained for 1,500 steps.

Because the no-position model used a different training duration, comparisons involving that model should be interpreted cautiously.

### Main Observation

Under the configuration used in this experiment, the RoPE model achieved a lower validation loss and validation perplexity than the learned-position baseline.

The validation loss decreased from:

- Learned Position: `1.8867`
- RoPE: `1.6090`

The validation perplexity decreased from:

- Learned Position: `6.5978`
- RoPE: `4.9981`

This result suggests that RoPE was beneficial for this particular compact character-level language modeling experiment.

The result should not be interpreted as proof that RoPE will outperform learned positional embeddings across all datasets, architectures, or training settings.

---

## Parameter Comparison

| Model | Parameter Count |
|---|---:|
| Learned Position | 897,664 |
| RoPE | 881,280 |
| No Position | 881,280 |

The learned-position model contains additional trainable parameters because it uses a learned positional embedding table.

The RoPE and no-position models do not require this learned positional embedding table.

---

## Qualitative Generation

The notebook includes text generation experiments using different sampling temperatures.

The generated samples demonstrate that the models learn some character-level patterns, spacing, punctuation, and WikiText-like formatting.

However, the outputs contain malformed words and incoherent sequences.

This is expected given the following factors:

- Compact model architecture.
- Character-level tokenization.
- Limited training duration.
- Limited context length.
- Absence of large-scale pretraining.
- Small experimental compute budget.

The generation examples are therefore used to inspect qualitative model behavior rather than to demonstrate production-level text generation.

---

## Comparison with the Original Paper

This project is inspired by the original RoFormer paper but is **not an exact reproduction** of its experiments.

### Similarities

The implementation shares several core ideas with the paper:

- RoPE is applied to query and key representations.
- Position information is incorporated through rotations.
- Sine and cosine values are used to implement the rotation efficiently.
- The model uses causal self-attention.
- The project compares a positional encoding approach with a baseline.
- The project investigates how positional representations affect Transformer behavior.

### Differences

| Aspect | This Project | Original RoFormer Paper |
|---|---|---|
| Main objective | Educational implementation and comparison | Research evaluation of RoPE and RoFormer |
| Dataset | WikiText-2 | Multiple datasets and NLP benchmarks |
| Tokenization | Character-level | Task-dependent tokenization |
| Language modeling task | Causal next-character prediction | Includes pretraining and other language modeling settings |
| Model size | Compact Transformer | Larger research architectures |
| Context length | 128 | Varies across experiments |
| Training scale | Small Colab-scale experiment | Large-scale experiments |
| Evaluation | Loss, perplexity, generation | BLEU, MLM loss, GLUE metrics, long-text evaluation, and others |
| Hardware | Google Colab-compatible setup | Experiments conducted using multiple V100 GPUs |
| Reproduction status | Educational adaptation | Original research implementation |

The original paper evaluates RoPE in machine translation, language model pretraining, GLUE tasks, linear attention, and Chinese long-text settings.

Its reported experiments use substantially larger datasets, models, and training configurations than those used in this project.

Consequently, the results in this repository should be understood as an independent small-scale investigation rather than a direct replication of the paper's reported benchmarks.

---

## Limitations

Several limitations should be considered when interpreting the results.

### 1. Small Model and Dataset Configuration

The experiment uses a compact Transformer and character-level tokenization. Its behavior may not generalize to larger models or subword-tokenized language models.

### 2. Limited Training Duration

The models were trained for a relatively small number of steps compared with large-scale language model pretraining.

### 3. Different Ablation Training Schedules

The learned-position and RoPE models were trained for 3,000 steps, whereas the no-position model was trained for 1,500 steps.

A fairer ablation would use identical training schedules and freshly initialized models.

### 4. Single Experimental Configuration

The results are based on one primary model configuration and do not establish how RoPE behaves across different model sizes, context lengths, datasets, or random seeds.

### 5. Limited Reproducibility Scope

Although the notebook records the main configuration and evaluation results, more extensive experiments would be required to measure variance across multiple random seeds.

---

## What I Learned

Through this project, I developed practical understanding of the following concepts:

- How causal self-attention operates.
- Why Transformers require positional information.
- The difference between absolute and relative positional representations.
- How RoPE encodes position through rotations.
- Why RoPE is applied to queries and keys rather than values in this implementation.
- How sine and cosine caches can make rotary transformations efficient.
- How causal masking prevents access to future tokens.
- How to construct a Transformer language model from basic components.
- How to implement and validate tensor operations using PyTorch.
- How to train a language model using cross-entropy loss.
- How perplexity can be derived from language modeling loss.
- How to compare model variants using controlled evaluation metrics.
- Why experimental fairness and training consistency matter.
- How to distinguish an educational reproduction from an exact research replication.
- How to document limitations and interpret experimental results responsibly.

---

## Future Improvements

The following improvements could extend this project:

### Experimental Improvements

- Train all models using identical training schedules.
- Repeat experiments across multiple random seeds.
- Compare RoPE with sinusoidal positional encoding.
- Compare RoPE with other relative positional encoding methods.
- Investigate different context lengths.
- Perform experiments with different model sizes.
- Evaluate the effect of different RoPE base frequencies.
- Analyze training convergence more systematically.
- Measure the impact of RoPE on longer sequences.

### Modeling Improvements

- Introduce subword tokenization.
- Increase the Transformer depth and embedding dimension.
- Train for a larger number of steps.
- Experiment with learning-rate schedules.
- Add mixed-precision training.
- Use more efficient attention implementations.
- Investigate longer-context language modeling.

### Evaluation Improvements

- Evaluate multiple random seeds.
- Report confidence intervals or variation across runs.
- Compare models under equal parameter budgets.
- Compare models under equal computational budgets.
- Evaluate performance across multiple datasets.
- Investigate performance on long-range dependencies.
- Compare generated text using additional qualitative and quantitative metrics.

### Research Extensions

- Study the mathematical relationship between rotation and relative position.
- Analyze the long-term decay property described in the original paper.
- Implement RoPE in a linear-attention architecture.
- Investigate the effect of rotary embeddings on attention patterns.
- Compare different rotary frequency schedules.
- Explore RoPE scaling methods for longer context lengths.

---

## Project Structure

```text
rotary-position-embedding-study/
│
├── notebooks/
│   └── roformer_rope_study.ipynb
│
├── results/
│   ├── training_curves.png
│   └── evaluation_results.md
│
├── README.md
├── summary.md
├── requirements.txt
└── .gitignore
```

The exact structure may vary depending on how the notebook and generated results are organized.

---

## Reproducibility

The notebook includes the primary model configuration, training settings, evaluation procedure, and experiment outputs.

For a more rigorous reproduction, future experiments should additionally record:

- Random seeds for each experiment.
- Exact library versions.
- Hardware configuration.
- Training duration.
- Dataset preprocessing details.
- Checkpoint selection criteria.
- Multiple evaluation runs.

---

## Acknowledgements

This project is based on the ideas presented in:

> Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding."

Original paper:

https://arxiv.org/abs/2104.09864

Official implementation:

https://github.com/ZhuiyiTechnology/roformer
