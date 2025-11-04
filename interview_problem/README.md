# Transformer Debugging Interview Problem

## Problem Statement

You are given a simplified GPT (Generative Pre-trained Transformer) implementation that **should** be able to overfit on a tiny dataset, but currently fails to learn properly. Your task is to:

1. **Debug and fix all bugs** in the code
2. **Run the training script** on the provided small dataset
3. **Achieve overfitting** - the training loss should drop below 0.1

## Background

This is a minimal implementation of a transformer-based language model. The model should easily overfit on the small provided dataset (only ~2500 tokens), but there are **5 intentional bugs** that prevent proper training:

- **4 bugs in the transformer model** (`model.py`)
- **1 bug in the loss computation** (`train.py`)

All bugs are realistic mistakes that commonly occur in transformer implementations.

## Setup

### Files Structure
```
interview_problem/
├── README.md           # This file
├── train.py           # Training script (contains 1 bug)
├── model.py           # Transformer model (contains 4 bugs)
└── data/
    ├── train.bin      # Training data (~2500 tokens)
    ├── val.bin        # Validation data (~280 tokens)
    └── meta.pkl       # Vocabulary metadata
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ (recommended for better performance)
- NumPy

Install dependencies:
```bash
pip install torch numpy
```

### Dataset

The dataset is already preprocessed and ready to use. It's a small excerpt from Shakespeare (repeated for easier overfitting), with:
- **Vocabulary size**: 44 characters
- **Training tokens**: ~2,500
- **Validation tokens**: ~280

This tiny dataset should be trivial to overfit for a working model.

## Your Task

### Step 1: Understand the Code

Review both files:
- `model.py`: Contains the GPT model implementation
  - `LayerNorm`: Layer normalization
  - `CausalSelfAttention`: Multi-head self-attention with causal masking
  - `MLP`: Feed-forward network
  - `Block`: Transformer block (attention + MLP)
  - `GPT`: Main model class

- `train.py`: Training script with data loading and training loop

### Step 2: Run Initial Training

Run the training script to see the current behavior:

```bash
cd interview_problem
python train.py
```

You should observe that:
- The model trains but **loss doesn't decrease properly**
- Training loss remains high (>2.0)
- The model fails to overfit

### Step 3: Debug and Fix

Find and fix all 5 bugs. Tips:
- Read the comments marked with "BUG" - they indicate where bugs are
- Think about common mistakes in transformer implementations
- Consider:
  - Attention mechanism (scoring, scaling, what matrices are used)
  - Residual connections
  - Layer normalization placement
  - Loss computation and scaling

### Step 4: Verify Success

After fixing all bugs, run training again:

```bash
python train.py
```

**Success criteria:**
- Training loss should drop below **0.1** (ideally < 0.05)
- The model should clearly overfit to the training data
- This should happen within 1000-2000 iterations

## Hints

<details>
<summary>Click for hints if you're stuck</summary>

### Hint 1: Bug Locations
- 2 bugs are in the `CausalSelfAttention` class
- 2 bugs are in the `Block` class
- 1 bug is in the training loop in `train.py`

### Hint 2: Bug Types
Think about:
- Is the attention mechanism computing scores correctly?
- Are attention scores properly scaled?
- Are residual connections present where they should be?
- Is LayerNorm applied at the right time?
- Is the loss computed correctly?

### Hint 3: Common Transformer Mistakes
- Forgetting to scale attention scores by √d_k
- Using wrong matrices in attention computation
- Missing or incorrect residual connections
- Wrong normalization order (pre-norm vs post-norm)
- Incorrect loss scaling/reduction

</details>

## Expected Behavior

### Before Fixes (with bugs):
```
Step    0 | train loss 3.8234 | val loss 3.8156 | time 0.12s
Step  100 | train loss 3.2145 | val loss 3.2089 | time 1.45s
Step  200 | train loss 3.1234 | val loss 3.1156 | time 1.52s
...
Final train loss: 2.8945
✗ Training loss is still high. There may be bugs preventing learning.
```

### After Fixes (bugs fixed):
```
Step    0 | train loss 3.7891 | val loss 3.7823 | time 0.11s
Step  100 | train loss 1.2345 | val loss 1.4567 | time 1.43s
Step  200 | train loss 0.3456 | val loss 0.8901 | time 1.48s
Step  300 | train loss 0.0823 | val loss 0.7234 | time 1.51s
...
Final train loss: 0.0234
✓ SUCCESS! Model has overfit to training data (loss < 0.1)
```

## Architecture Details

### Model Configuration
- **Layers**: 4 transformer blocks
- **Attention heads**: 4
- **Embedding dimension**: 128
- **Block size**: 64 tokens
- **Parameters**: ~0.5M

### Training Configuration
- **Batch size**: 16
- **Learning rate**: 1e-3
- **Iterations**: 2000
- **Optimizer**: AdamW (no weight decay)

## Debugging Tips

1. **Add print statements** to verify tensor shapes
2. **Check intermediate values** - are attention scores reasonable?
3. **Monitor gradients** - are they flowing properly?
4. **Compare with reference implementations** - how should attention work?
5. **Test components individually** - does each layer do what you expect?

## Time Estimate

- **Understanding code**: 10-15 minutes
- **Finding bugs**: 20-30 minutes
- **Testing fixes**: 10-15 minutes
- **Total**: ~45-60 minutes

## Evaluation Criteria

You will be evaluated on:
1. **Correctness**: All bugs found and fixed properly
2. **Understanding**: Can explain what each bug was and why it mattered
3. **Debugging process**: Systematic approach to finding issues
4. **Final result**: Model successfully overfits (train loss < 0.1)

Good luck! 🚀
