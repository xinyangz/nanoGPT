"""
Quick test script to verify bugs prevent learning and fixes enable learning.
Runs for only 300 iterations to save time.
"""

import os
import time
import pickle
import numpy as np
import torch

# Test both versions
print("=" * 80)
print("TESTING BUGGY VERSION")
print("=" * 80)

from model import GPT as BuggyGPT, GPTConfig as BuggyConfig

# Setup
torch.manual_seed(1337)
device = 'cpu'  # Use CPU for faster testing
data_dir = 'data'
batch_size = 16
block_size = 64

# Load metadata and data
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')


def get_batch():
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# Test buggy model
config = BuggyConfig(n_layer=4, n_head=4, n_embd=128, block_size=block_size,
                     dropout=0.0, vocab_size=vocab_size, bias=True)
model = BuggyGPT(config)
model.to(device)
optimizer = model.configure_optimizers(weight_decay=0.0, learning_rate=1e-3)

losses_buggy = []
for i in range(300):
    X, Y = get_batch()
    logits, loss = model(X, Y)
    loss = loss * block_size  # BUG #5 present
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if i % 50 == 0:
        print(f"Step {i:3d} | loss {loss.item():.4f}")
        losses_buggy.append(loss.item())

print(f"Buggy version final loss: {losses_buggy[-1]:.4f}")
print()

# Test fixed model
print("=" * 80)
print("TESTING FIXED VERSION")
print("=" * 80)

from model_fixed import GPT as FixedGPT, GPTConfig as FixedConfig

torch.manual_seed(1337)  # Same seed for fair comparison

config = FixedConfig(n_layer=4, n_head=4, n_embd=128, block_size=block_size,
                     dropout=0.0, vocab_size=vocab_size, bias=True)
model = FixedGPT(config)
model.to(device)
optimizer = model.configure_optimizers(weight_decay=0.0, learning_rate=1e-3)

losses_fixed = []
for i in range(300):
    X, Y = get_batch()
    logits, loss = model(X, Y)
    # No bug #5 - use loss as-is
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if i % 50 == 0:
        print(f"Step {i:3d} | loss {loss.item():.4f}")
        losses_fixed.append(loss.item())

print(f"Fixed version final loss: {losses_fixed[-1]:.4f}")
print()

# Compare
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Buggy version: {losses_buggy[0]:.4f} → {losses_buggy[-1]:.4f} (Δ {losses_buggy[0] - losses_buggy[-1]:.4f})")
print(f"Fixed version: {losses_fixed[0]:.4f} → {losses_fixed[-1]:.4f} (Δ {losses_fixed[0] - losses_fixed[-1]:.4f})")
print()

if losses_fixed[-1] < losses_buggy[-1] - 0.5:
    print("✓ Fixed version learns significantly better!")
else:
    print("✗ Warning: Fixed version not learning much better. Check implementation.")
