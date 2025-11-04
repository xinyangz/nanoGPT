# Testing Instructions

This document explains how to verify the interview problem setup works correctly.

## Requirements

```bash
pip install torch numpy
```

## Quick Test (Recommended)

Run the quick test script that compares buggy vs fixed versions:

```bash
cd interview_problem
python quick_test.py
```

**Expected output:**
- Buggy version: Loss stays high or learns very slowly
- Fixed version: Loss decreases significantly
- Clear difference showing fixed version learns better

## Full Test - Buggy Version

Test that the buggy version fails to learn:

```bash
python train.py
```

**Expected behavior:**
- Training runs for 2000 iterations
- Loss stays high (>2.0) or decreases very slowly
- Final message: "✗ Training loss is still high. There may be bugs preventing learning."
- **This confirms the bugs prevent learning**

## Full Test - Fixed Version

Test that the fixed version successfully overfits:

```bash
python train_fixed.py
```

**Expected behavior:**
- Training runs for 2000 iterations
- Loss decreases rapidly
- Final train loss < 0.1 (often 0.02-0.05)
- Final message: "✓ SUCCESS! Model has overfit to training data (loss < 0.1)"
- **This confirms the fixes enable learning**

## What Each Bug Does

### Bug 1: Missing Attention Scaling
- **Impact**: Attention weights become too sharp (softmax saturation)
- **Observable**: Gradients become very small, learning is slow
- **Severity**: High - significantly impairs learning

### Bug 2: Wrong Matrix in Attention (k instead of v)
- **Impact**: Attention aggregates keys instead of values
- **Observable**: Model cannot properly propagate information
- **Severity**: Critical - breaks attention mechanism semantics

### Bug 3: Missing Residual After Attention
- **Impact**: Gradients cannot flow directly backward
- **Observable**: Vanishing gradients, no learning in deeper layers
- **Severity**: Critical - prevents gradient flow

### Bug 4: Missing Residual After MLP
- **Impact**: Same as Bug 3, but for MLP layers
- **Observable**: Vanishing gradients through feed-forward layers
- **Severity**: Critical - prevents gradient flow

### Bug 5: Loss Scaling in Training Loop
- **Impact**: Loss is scaled by block_size (64x too large!)
- **Observable**: Gradients are 64x too large, causing instability
- **Severity**: High - prevents convergence

## Combined Effect

With all 5 bugs present:
- Attention mechanism is broken (Bugs 1, 2)
- Gradient flow is blocked (Bugs 3, 4)
- Loss computation is wrong (Bug 5)
- **Result**: Model cannot learn at all

## Partial Fixes

Interesting to test fixing bugs incrementally:

### Fix only Bug 5 (loss scaling)
```python
# In train.py, line ~118, change:
loss = loss * block_size  # BUGGY
# to:
loss = loss  # FIXED
```
**Result**: Still won't learn well due to bugs 1-4

### Fix Bugs 1, 2 (attention)
```python
# In model.py, CausalSelfAttention.forward()

# Line ~67, add scaling:
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

# Line ~75, use v instead of k:
y = att @ v
```
**Result**: Some learning, but still blocked by missing residuals

### Fix Bugs 3, 4 (residuals)
```python
# In model.py, Block.forward()

# Line ~115:
x = x + self.attn(self.ln_1(x))  # Add residual

# Line ~120:
x = x + self.mlp(self.ln_2(x))  # Add residual
```
**Result**: Much better, but still affected by loss scaling bug

### Fix All Bugs
**Result**: ✓ Rapid overfitting, training loss < 0.1

## Debugging Tips for Candidates

If a candidate is stuck, suggest they:

1. **Check tensor shapes**: Print shapes at each step
2. **Verify attention scores**: Are they reasonable values?
3. **Check gradients**: Are they flowing? (`model.parameters()` after `loss.backward()`)
4. **Test components**: Run individual modules in isolation
5. **Compare with reference**: Look at transformer diagrams/papers
6. **Think about principles**: What makes transformers work?

## Time Estimates

- Quick test: ~2-3 minutes runtime
- Full buggy test: ~5-10 minutes runtime
- Full fixed test: ~5-10 minutes runtime
- Total testing time: ~15-20 minutes

## Hardware Requirements

- **CPU**: Works fine, slightly slower (~10 min per full run)
- **GPU**: Faster (~3-5 min per full run)
- **RAM**: ~500MB sufficient
- **Disk**: <10MB for dataset

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
```

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install numpy
```

### Data files missing
```bash
cd interview_problem/data
python prepare_tiny_data.py
```

### Both versions fail to learn
- Check PyTorch installation
- Verify data files exist (train.bin, val.bin, meta.pkl)
- Try running on CPU if GPU issues

### Both versions learn equally well
- Verify you're testing the right files (model.py vs model_fixed.py)
- Check that bugs are actually present in model.py
- Ensure train.py has the loss scaling bug

## Success Criteria

The setup is working correctly if:
1. ✓ Buggy version fails to achieve loss < 0.5
2. ✓ Fixed version achieves loss < 0.1
3. ✓ Clear difference between buggy and fixed versions
4. ✓ All 5 bugs are present and documented
5. ✓ All 5 fixes make the model train successfully
