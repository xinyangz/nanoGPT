# Solutions to Transformer Debugging Problem

## Summary of Bugs

This document describes all 5 bugs and their fixes. **Candidates should not see this file during the interview.**

---

## Bug 1: Missing Attention Score Scaling

**Location**: `model.py`, line ~67, in `CausalSelfAttention.forward()`

**The Bug**:
```python
# BUGGY
att = (q @ k.transpose(-2, -1))  # No scaling!
```

**The Fix**:
```python
# CORRECT
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

**Why it matters**:
- Attention scores are computed as Q·K^T
- Without scaling by √d_k (where d_k is the dimension per head), the dot products can become very large
- Large values cause the softmax to saturate, producing very sharp distributions (all weight on one position)
- This makes gradients very small and prevents learning
- This is one of the key innovations in the "Attention is All You Need" paper

**Impact**: High - prevents proper gradient flow through attention mechanism

---

## Bug 2: Using Wrong Matrix in Attention

**Location**: `model.py`, line ~75, in `CausalSelfAttention.forward()`

**The Bug**:
```python
# BUGGY - using k instead of v!
y = att @ k
```

**The Fix**:
```python
# CORRECT
y = att @ v
```

**Why it matters**:
- Attention mechanism has three components: Query (Q), Key (K), Value (V)
- Q and K are used to compute attention weights (which positions to attend to)
- V contains the actual content to aggregate
- The formula is: Attention(Q,K,V) = softmax(QK^T / √d_k) · V
- Using K instead of V means we're aggregating the key representations instead of values
- This completely breaks the attention mechanism's ability to propagate information

**Impact**: Critical - completely breaks attention semantics

---

## Bug 3: Missing Residual Connection After Attention

**Location**: `model.py`, line ~115, in `Block.forward()`

**The Bug**:
```python
# BUGGY - no residual!
x = self.attn(self.ln_1(x))
```

**The Fix**:
```python
# CORRECT - add residual
x = x + self.attn(self.ln_1(x))
```

**Why it matters**:
- Residual connections (skip connections) are essential in deep networks
- They allow gradients to flow directly backward through the network
- Without them, gradients vanish in deep networks, preventing learning
- ResNet showed that residuals enable training of very deep networks
- In transformers, residual connections are critical for stacking multiple layers

**Impact**: Critical - prevents gradient flow, especially in deeper networks

---

## Bug 4: Missing Residual Connection After MLP

**Location**: `model.py`, line ~120, in `Block.forward()`

**The Bug**:
```python
# BUGGY - no residual!
x = self.mlp(self.ln_2(x))
```

**The Fix**:
```python
# CORRECT - add residual
x = x + self.mlp(self.ln_2(x))
```

**Why it matters**:
- Same as Bug 3, but for the MLP (feed-forward) component
- Every transformer block should have TWO residual connections:
  1. Around the attention sublayer
  2. Around the MLP sublayer
- Without both residuals, the network cannot learn effectively
- This is standard in all transformer architectures (BERT, GPT, T5, etc.)

**Impact**: Critical - prevents gradient flow through the feed-forward layers

---

## Bug 5: Incorrect Loss Scaling

**Location**: `train.py`, line ~118, in training loop

**The Bug**:
```python
# BUGGY - multiplying by block_size!
loss = loss * block_size
```

**The Fix**:
```python
# CORRECT - use loss as-is
loss = loss
# Or simply remove the scaling line
```

**Why it matters**:
- The model's forward pass already returns the mean cross-entropy loss per token
- Multiplying by `block_size` scales the loss by 64x!
- This causes gradients to be 64x too large
- Large gradients lead to:
  - Optimizer taking steps that are too large
  - Training instability
  - Poor convergence
- The learning rate is tuned assuming properly scaled gradients

**Impact**: High - causes training instability and prevents convergence

---

## Expected Behavior

### With All Bugs (Baseline)
- Loss stays high (~3.0+) and barely decreases
- Model fails to learn anything meaningful
- Cannot overfit even on tiny dataset

### Fixing Only Attention Bugs (#1, #2)
- Some learning occurs but still limited
- Loss might decrease to ~1.5-2.0
- Still cannot overfit properly

### Fixing Attention + Residual Bugs (#1, #2, #3, #4)
- Significant learning occurs
- Loss decreases to ~0.5-1.0
- Still suboptimal due to gradient scaling issue

### Fixing All Bugs (#1, #2, #3, #4, #5)
- ✓ Rapid learning and overfitting
- Training loss drops below 0.1 (often to 0.01-0.05)
- Model perfectly memorizes the training data
- This is the expected behavior!

---

## Testing the Fixes

### Quick Test
```bash
cd interview_problem
python train.py
```

Look for the final output:
- ✓ **Success**: "✓ SUCCESS! Model has overfit to training data (loss < 0.1)"
- ✗ **Failure**: "✗ Training loss is still high. There may be bugs preventing learning."

### Expected Training Curve (All Bugs Fixed)
```
Step    0 | train loss 3.78 | val loss 3.79
Step  100 | train loss 1.45 | val loss 1.89
Step  200 | train loss 0.67 | val loss 1.34
Step  300 | train loss 0.23 | val loss 0.98
Step  400 | train loss 0.08 | val loss 0.87
...
Final train loss: 0.02-0.05
```

---

## Common Debugging Approaches

Good candidates might:
1. **Start with the fundamentals**: Check attention mechanism implementation
2. **Add logging**: Print tensor shapes and intermediate values
3. **Verify gradients**: Check if gradients are flowing (use `loss.backward()` and inspect `param.grad`)
4. **Test components**: Run individual modules in isolation
5. **Compare with references**: Look up transformer architecture diagrams
6. **Think about principles**: What are the key components that make transformers work?

---

## Grading Rubric

### Excellent (A)
- Finds all 5 bugs independently
- Explains each bug clearly and correctly
- Demonstrates systematic debugging approach
- Achieves target loss (<0.1) quickly

### Good (B)
- Finds 4-5 bugs with minimal hints
- Understands the impact of each bug
- Uses reasonable debugging strategies
- Achieves target loss with some iteration

### Fair (C)
- Finds 3-4 bugs with hints
- Can explain most bugs when prompted
- Eventually achieves target loss

### Needs Improvement (D)
- Finds fewer than 3 bugs even with hints
- Struggles to explain bug impacts
- Does not achieve target loss

---

## Time Expectations

- **15 min**: Quick candidates might finish in 15-20 minutes
- **30 min**: Average candidates finish in 30-45 minutes
- **60 min**: This is the maximum time budget
- **60+ min**: Consider providing more direct hints

---

## Additional Discussion Questions

After the candidate solves the problem, you can ask:

1. **Architecture**: "How would you modify this for a encoder-decoder model?"
2. **Optimization**: "What are other common transformer optimizations?"
3. **Scaling**: "How would you handle longer sequences?"
4. **Variants**: "What's the difference between GPT and BERT?"
5. **Production**: "What would you add to make this production-ready?"

---

## Files Summary

- `model.py` - Contains Bugs #1, #2, #3, #4
- `train.py` - Contains Bug #5
- `data/` - Preprocessed tiny dataset (ready to use)
- `README.md` - Candidate instructions
- `SOLUTIONS.md` - This file (for interviewer only)
