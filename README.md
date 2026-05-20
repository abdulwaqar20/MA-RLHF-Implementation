# MA-PPO for RLHF: Macro Action Reinforcement Learning from Human Feedback

Implementation of a Macro Action PPO (MA-PPO) approach for Reinforcement Learning from Human Feedback (RLHF), inspired by the **MA-RLHF: Macro Action Reinforcement Learning from Human Feedback** paper.

This project compares:

- Standard PPO using token-level actions (`n = 1`)
- Macro Action PPO (MA-PPO) using grouped token actions (`n = 5`)

The implementation uses GPT-2 for text generation and evaluates performance on summarization tasks.

---

## Overview

Traditional RLHF methods typically optimize language models at the token level, where each generated token represents an individual action.

Inspired by the MA-RLHF paper, this implementation introduces **macro actions**, where multiple tokens are grouped into a single action. Instead of optimizing every token independently, optimization occurs over groups of tokens.

The motivation behind macro actions includes:

- Reduced action space complexity
- Faster convergence
- Improved training efficiency
- Better long-range decision making

---

## Methodology

### Standard PPO (Baseline)

Token-level action representation:

```

aₜ = generated token

```

Characteristics:

- Individual token optimization
- Large action sequence length
- Higher computational overhead
- Fine-grained policy updates

---

### MA-PPO (Proposed Method)

Macro action representation:

```

Aₜ = {a₁,a₂,...,aₙ}

```

where:

```

n = 5

```

Macro action probability:

```

P(Aₜ)= ∏ P(aᵢ)

```

In log-space:

```

log P(Aₜ)= Σ log P(aᵢ)

```

The implementation computes joint probabilities by summing token log probabilities and applies PPO optimization at the macro-action level.

---

## Features

- GPT-2 based language generation
- PPO baseline implementation
- Macro Action PPO implementation
- Reward-based summarization training
- Performance comparison
- Convergence analysis
- Reward visualization
- Training time comparison
- Macro action grouping

---

## Project Structure

```text
project/
│
├── RL_Research.ipynb
├── README.md
├── requirements.txt
├── outputs/
│   └── ppo_vs_mappo_comparison.png
```

---

## Dataset

Dataset used:

**CarperAI/OpenAI Summarization Comparisons**

Task:

- Text summarization

Dataset source:

https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons

Subset used in implementation:

```

500 samples

```

---

## Model

Base language model:

**GPT-2**

HuggingFace implementation:

```python
GPT2LMHeadModel.from_pretrained("gpt2")
```

Training strategy:

- Freeze most parameters
- Train only LM Head

This reduces memory requirements and speeds up experimentation.

---

## Reward Function

The reward function evaluates generated summaries based on:

### 1. Sentence Count Reward

Encourages concise summaries.

Target:

- 1–2 sentences

---

### 2. Word Count Reward

Encourages informative summaries.

Target:

- 15–50 words

---

### 3. Diversity Reward

Encourages lexical diversity.

Metric:

```python
unique_words / total_words
```

Overall reward:

```python
reward =
sentence_reward
+ word_reward
+ diversity_reward
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/MA-PPO-RLHF.git

cd MA-PPO-RLHF
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file:

```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.23.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.66.0
```

---

## Hardware Requirements

Minimum:

- 8 GB RAM
- GPU with 6 GB VRAM

Recommended:

- NVIDIA GPU with CUDA support
- 12–16 GB VRAM

Tested using:

- GPT-2
- CUDA enabled training

---

## Running the Project

Run notebook:

```bash
jupyter notebook RL_Research.ipynb
```

or

```bash
jupyter lab RL_Research.ipynb
```

---

## Training Pipeline

### PPO Training

1. Sample prompt
2. Generate response
3. Compute token log probabilities
4. Compute reward
5. Calculate PPO loss
6. Update policy

---

### MA-PPO Training

1. Sample prompt
2. Generate response
3. Compute token log probabilities
4. Group tokens into macro actions
5. Compute macro probabilities
6. Compute reward
7. Calculate PPO loss on macro actions
8. Update policy

---

## Evaluation Metrics

The implementation compares:

- Average reward
- Convergence speed
- Policy loss
- Training time
- Reward distribution

---

## Generated Visualizations

The notebook generates:

### Reward Comparison

Shows learning progress between:

- PPO
- MA-PPO

### Loss Convergence

Shows policy optimization behavior.

### Reward Distribution

Displays reward score distributions.

### Final Performance Comparison

Bar chart of final rewards.

Generated output:

```text
outputs/
└── ppo_vs_mappo_comparison.png
```

---

## Expected Results

Typical observations from experiments:

- Faster convergence using macro actions
- Reduced training time
- Higher average reward
- More stable learning behavior

Performance may vary depending on:

- Random initialization
- Dataset size
- Reward function
- Macro action size

---

## Hyperparameters

Current implementation:

```python
learning_rate = 1e-4

macro_n = 5

num_steps = 50

max_generation_length = 40

clip_epsilon = 0.2
```

---

## Limitations

Current implementation is a research prototype and includes several simplifications:

- Uses heuristic rewards rather than learned reward models
- Random advantages instead of value estimation
- GPT-2 only
- Small-scale experiments
- Limited training steps

Future improvements:

- Train a reward model
- Add value function estimation
- Scale to larger LLMs
- Dynamic macro action sizes
- Human preference datasets

---

## Reference

MA-RLHF paper:

```bibtex
@article{ma_rlhf,
title={MA-RLHF: Macro Action Reinforcement Learning from Human Feedback},
author={Authors},
year={2025}
}
```

---

## Acknowledgments

- MA-RLHF paper for macro action ideas
- HuggingFace Transformers
- OpenAI summarization datasets
- PyTorch

---

## License

MIT License
