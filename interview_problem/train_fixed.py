"""
FIXED Training Script - For testing purposes only
This version uses the fixed model for comparison.
"""

import os
import time
import pickle
import numpy as np
import torch
from model_fixed import GPT, GPTConfig

# Training Configuration
data_dir = 'data'
batch_size = 16
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
max_iters = 2000
learning_rate = 1e-3
weight_decay = 0.0
eval_interval = 100
eval_iters = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Setup
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) if device == 'cuda' else torch.nullcontext()

# Load metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
print(f"Vocabulary size: {vocab_size}")

# Load data
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

print(f"Train data: {len(train_data)} tokens")
print(f"Val data: {len(val_data)} tokens")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Model Initialization
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
    vocab_size=vocab_size,
    bias=True,
)

config = GPTConfig(**model_args)
model = GPT(config)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate)

if hasattr(torch, 'compile'):
    print("Compiling model...")
    model = torch.compile(model)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Training Loop
print(f"\nStarting training for {max_iters} iterations...")
print("=" * 80)

t0 = time.time()
best_val_loss = float('inf')

for iter_num in range(max_iters):

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter_num:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | "
              f"time {(time.time() - t0):.2f}s")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']

        t0 = time.time()

    X, Y = get_batch('train')

    with ctx:
        logits, loss = model(X, Y)
        # FIXED: No incorrect scaling

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

# Final Evaluation
print("=" * 80)
print("Training complete!")
losses = estimate_loss()
print(f"Final train loss: {losses['train']:.4f}")
print(f"Final val loss: {losses['val']:.4f}")
print(f"Best val loss: {best_val_loss:.4f}")

if losses['train'] < 0.1:
    print("\n✓ SUCCESS! Model has overfit to training data (loss < 0.1)")
else:
    print(f"\n✗ Training loss is still high ({losses['train']:.4f}).")
