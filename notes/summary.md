
# Project Summary: Rotary Position Embedding Study

## Project Title

**An Empirical Study of Rotary Position Embedding in Compact Transformers**

## Overview

This project presents a from-scratch educational implementation of Rotary Position Embedding (RoPE), a positional encoding method introduced in the RoFormer research paper.

The objective was to understand how positional information can be incorporated into Transformer self-attention and to empirically compare RoPE with learned positional embeddings and a no-position ablation.

A compact character-level causal Transformer was implemented and trained on the WikiText-2 dataset using PyTorch. The project covers the implementation of Transformer components, rotary position encoding, training, evaluation, text generation, and ablation analysis.

## Research Motivation

Self-attention does not inherently encode the sequential order of tokens. Positional representations are therefore important for enabling Transformer models to capture sequence structure.

RoPE introduces position-dependent rotations to query and key representations. This provides a mechanism for encoding absolute position while allowing relative position information to emerge within the attention inner product.

This project investigates the behavior of RoPE in a small-scale language modeling setting.

## Methodology

The project implements three model variants:

1. **Learned Position Transformer:** Uses trainable positional embeddings added to token embeddings.
2. **RoPE Transformer:** Applies rotary position encoding to query and key representations within self-attention.
3. **No Position Transformer:** Removes explicit positional encoding as an ablation.

The models use a compact causal Transformer architecture with:

- Character-level tokenization.
- Context length of 128.
- Embedding dimension of 128.
- Four attention heads.
- Three Transformer layers.
- Causal self-attention.
- AdamW optimization.
- Cross-entropy language modeling loss.

## Dataset

The models are trained on the raw WikiText-2 dataset obtained through the Hugging Face Datasets library.

The dataset is processed into character-level sequences for next-character prediction.

Dataset link:

https://huggingface.co/datasets/Salesforce/wikitext

## Results

The final evaluation results were:

| Model | Validation Loss | Validation Perplexity |
|---|---:|---:|
| Learned Position | 1.8867 | 6.5978 |
| RoPE | 1.6090 | 4.9981 |
| No Position | 2.3836 | 10.8437 |

The RoPE model achieved lower validation loss and perplexity than the learned-position baseline in this experiment.

The results indicate that RoPE was effective under the selected dataset, architecture, and training configuration. However, these findings are specific to the experimental setup and should not be generalized to all Transformer architectures or datasets.

The no-position ablation was trained for fewer steps than the other two models. Therefore, its result should be interpreted cautiously and does not represent a fully controlled comparison.

## Comparison with the Original Paper

This implementation is inspired by the RoFormer paper but is not an exact reproduction.

The original paper evaluates RoPE across several settings, including machine translation, language model pretraining, GLUE tasks, linear attention, and Chinese long-text tasks.

In contrast, this project uses:

- A compact Transformer.
- Character-level tokenization.
- WikiText-2.
- Causal next-character prediction.
- A limited training budget.
- Google Colab-compatible configurations.

The project focuses on understanding and implementing the central RoPE mechanism rather than reproducing the paper's full-scale experiments.

## Key Learning Outcomes

This project strengthened my understanding of:

- Transformer self-attention.
- Causal attention masking.
- Absolute and relative positional encoding.
- Rotary position embedding.
- Query-key transformations in attention.
- Efficient sine and cosine cache construction.
- PyTorch tensor manipulation.
- Causal language modeling.
- Cross-entropy loss and perplexity.
- Training and validation evaluation.
- Model ablation and experimental comparison.
- Research reproducibility and responsible interpretation of results.

A key takeaway was that implementing a research method requires more than reproducing its equations. It also requires understanding the assumptions, experimental setup, evaluation methodology, and limitations of the original work.

## Limitations

The project uses a small model, a limited training duration, and a single primary configuration.

The comparison is exploratory rather than statistically conclusive. The no-position model was trained for fewer steps, and the experiments were not repeated across multiple random seeds.

The generated samples also contain malformed words and incoherent sequences, reflecting the limited scale and training duration of the model.

## Future Work

Potential extensions include:

- Conducting fair comparisons with identical training schedules.
- Repeating experiments across multiple random seeds.
- Comparing different positional encoding techniques.
- Testing different context lengths and model sizes.
- Investigating the effect of RoPE frequency parameters.
- Using subword tokenization.
- Evaluating longer-range dependencies.
- Studying RoPE in linear-attention architectures.
- Investigating RoPE scaling methods for long-context modeling.

## Research Relevance

This project demonstrates practical experience in implementing and analyzing a research-inspired deep learning method.

It combines theoretical understanding with hands-on experimentation in:

- Natural language processing.
- Transformer architectures.
- Attention mechanisms.
- Representation learning.
- Experimental design.
- Model evaluation.
- Technical documentation.

The project was developed as a learning-oriented implementation and provides a foundation for further investigation into positional encoding methods and long-context Transformer architectures.

## References

Su, Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding."

Paper: https://arxiv.org/abs/2104.09864

Official implementation: https://github.com/ZhuiyiTechnology/roformer

Dataset: https://huggingface.co/datasets/Salesforce/wikitext
