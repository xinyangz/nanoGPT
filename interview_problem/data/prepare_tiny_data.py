"""
Prepare a tiny dataset for debugging interview.
This creates a very small dataset (a few thousand tokens) that should be easy to overfit.
"""
import numpy as np
import pickle

# A small sample text - repeated patterns for easy overfitting
text = """ROMEO:
Good morrow to you both. What counterfeit did I give you?

MERCUTIO:
The slip, sir, the slip; can you not conceive?

ROMEO:
Pardon, good Mercutio, my business was great; and in
such a case as mine a man may strain courtesy.

MERCUTIO:
That's as much as to say, such a case as yours
constrains a man to bow in the hams.

ROMEO:
Meaning, to curtsy.

MERCUTIO:
Thou hast most kindly hit it.

ROMEO:
A most courteous exposition.

MERCUTIO:
Nay, I am the very pink of courtesy.

ROMEO:
Pink for flower.

MERCUTIO:
Right.

ROMEO:
Why, then is my pump well flowered.
""" * 5  # Repeat 5 times to make it easier to overfit

# Get all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Total characters: {len(text)}")

# Create character to index mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode the text
data = np.array([stoi[c] for c in text], dtype=np.uint16)

# Simple split: 90% train, 10% val
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print(f"Train tokens: {len(train_data)}")
print(f"Val tokens: {len(val_data)}")

# Save to binary files
train_data.tofile('train.bin')
val_data.tofile('val.bin')

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("Dataset prepared successfully!")
print(f"Files created: train.bin ({len(train_data)} tokens), val.bin ({len(val_data)} tokens), meta.pkl")
