# Frankenstein LSTM text generation - complete implementation
# Run in order in a notebook cell or save as a script.

import numpy as np
import torch
torch.manual_seed(1)  # set random seed -- do not change!
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Task 1: Load the text file
# ---------------------------
file_path = "datasets/frankenstein.txt"  # change if different
with open(file_path, 'r', encoding='utf-8') as f:
    frankenstein = f.read()

print("Length of full file (chars):", len(frankenstein))

# ---------------------------
# Task 2: Slice out first letter
# ---------------------------
# Slice from index 1380 (inclusive) to 8230 (exclusive)
first_letter_text = frankenstein[1380:8230]
print("\n--- Extracted first letter preview (first 400 chars) ---\n")
print(first_letter_text[:400])
print("\nLength of first_letter_text:", len(first_letter_text))

# ---------------------------
# Task 3: Tokenize (character tokens)
# ---------------------------
tokenized_text = list(first_letter_text)
print("\nNumber of tokens in tokenized_text:", len(tokenized_text))

# ---------------------------
# Task 4: Create sorted unique tokens
# ---------------------------
unique_char_tokens = sorted(list(set(tokenized_text)))
print("\nUnique characters (sample 40):", unique_char_tokens[:40])

# ---------------------------
# Task 5: Create c2ix (char -> index)
# ---------------------------
c2ix = {c: i for i, c in enumerate(unique_char_tokens)}
print("\nVocabulary (char -> id) sample (first 30):")
for i, (ch, idx) in enumerate(c2ix.items()):
    if i < 30:
        print(repr(ch), ":", idx)
    else:
        break

# ---------------------------
# Task 6: Vocabulary size
# ---------------------------
vocab_size = len(c2ix)
print("\nVocab size:", vocab_size)

# ---------------------------
# Task 7: Create ix2c (index -> char)
# ---------------------------
ix2c = {i: c for c, i in c2ix.items()}

# ---------------------------
# Task 8: Map tokenized_text -> token ids
# ---------------------------
tokenized_id_text = [c2ix[c] for c in tokenized_text]
print("\nFirst 20 token ids:", tokenized_id_text[:20])

# ---------------------------
# Task 9: (import done above)
# ---------------------------

# ---------------------------
# Task 10: Define TextDataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, tokenized_text, seq_length):
        self.tokenized_text = tokenized_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokenized_text) - self.seq_length

    def __getitem__(self, idx):
        # features: seq_length tokens starting at idx
        features = torch.tensor(self.tokenized_text[idx: idx + self.seq_length], dtype=torch.long)
        # labels: next tokens (shifted by 1)
        labels = torch.tensor(self.tokenized_text[idx + 1: idx + 1 + self.seq_length], dtype=torch.long)
        return features, labels

# ---------------------------
# Task 11: seq_length = 48 and dataset
# ---------------------------
seq_length = 48
dataset = TextDataset(tokenized_id_text, seq_length)
print("\nNumber of sequences in dataset:", len(dataset))

# quick sample
sample_feat, sample_lab = dataset[0]
print("\nSample features shape:", sample_feat.shape, "Sample labels shape:", sample_lab.shape)
print("Sample features (chars):", ''.join(ix2c[idx.item()] for idx in sample_feat[:60]))

# ---------------------------
# Task 12: DataLoader
# ---------------------------
batch_size = 36
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print("\nDataloader created. Batches per epoch:", len(dataloader))

# ---------------------------
# Task 13: import nn done above
# ---------------------------

# ---------------------------
# Task 14: CharacterLSTM class
# ---------------------------
class CharacterLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=48, hidden_size=96):
        super(CharacterLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, states):
        # x: (batch, seq_len)
        x = self.embedding(x)  # -> (batch, seq_len, embedding_dim)
        out, states = self.lstm(x, states)  # out: (batch, seq_len, hidden_size)
        out = self.linear(out)  # (batch, seq_len, vocab_size)
        # reshape so it is (batch*seq_len, vocab_size) to be compatible with CrossEntropyLoss
        out = out.reshape(-1, out.size(2))
        return out, states

    def init_state(self, batch_size):
        # shapes: (num_layers * num_directions, batch, hidden_size)
        hidden = torch.zeros(1, batch_size, 96)
        cell = torch.zeros(1, batch_size, 96)
        return (hidden, cell)

# ---------------------------
# Task 15: instantiate model
# ---------------------------
lstm_model = CharacterLSTM(vocab_size=vocab_size, embedding_dim=48, hidden_size=96)
print("\nLSTM model created.")

# ---------------------------
# Task 16: loss function
# ---------------------------
loss = nn.CrossEntropyLoss()
print("Loss function (CrossEntropyLoss) initialized.")

# ---------------------------
# Task 17: optimizer
# ---------------------------
optimizer = optim.Adam(lstm_model.parameters(), lr=0.015)
print("Optimizer (Adam) initialized with lr=0.015.")

# ---------------------------
# Task 18: Training loop (5 epochs)
# ---------------------------
num_epochs = 5
lstm_model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    batches = 0
    for features, labels in dataloader:
        optimizer.zero_grad()
        # initialize states for current batch
        states = lstm_model.init_state(features.size(0))
        outputs, states = lstm_model(features, states)
        # outputs shape: (batch*seq_len, vocab_size)
        # labels shape should be (batch*seq_len,) for CrossEntropyLoss
        loss_val = loss(outputs, labels.view(-1))
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()
        batches += 1

    avg_loss = total_loss / batches if batches > 0 else 0.0
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

# ---------------------------
# Task 19: starting prompt
# ---------------------------
starting_prompt = "You will rejoice to hear"
print("\nStarting prompt:", repr(starting_prompt))

# ---------------------------
# Task 20: tokenize starting prompt to IDs (as tensor of shape (1, seq_len_prompt))
# ---------------------------
tokenized_id_prompt = torch.tensor([[c2ix.get(ch, 0) for ch in starting_prompt]], dtype=torch.long)  # shape (1, L)
print("Tokenized id prompt shape:", tokenized_id_prompt.shape)
print("Tokenized id prompt:", tokenized_id_prompt)

# ---------------------------
# Task 21: set model to eval
# ---------------------------
lstm_model.eval()

# ---------------------------
# Task 22: Generate next 500 chars
# ---------------------------
num_generated_chars = 500

# We'll prime the model with the whole starting prompt to get an initial state
generated_text = starting_prompt  # will build this up

with torch.no_grad():
    # initialize states for batch size 1
    states = lstm_model.init_state(1)

    # Prime the LSTM by passing the entire prompt (so states reflect prompt)
    out, states = lstm_model(tokenized_id_prompt, states)
    # out corresponds to logits for every position in the prompt; we will use the last position as the starting prediction
    last_logits = out[-1]  # shape: (vocab_size,)

    # pick the most likely next char (argmax)
    next_id = torch.argmax(last_logits).item()
    next_char = ix2c[next_id]
    generated_text += next_char

    # now iteratively generate one char at a time
    cur_input = torch.tensor([[next_id]], dtype=torch.long)  # shape (1,1)
    for i in range(num_generated_chars - 1):  # we already generated 1 char
        out, states = lstm_model(cur_input, states)
        # out shape (1, vocab_size)
        logits = out[-1]  # get the last/time step logits
        next_id = torch.argmax(logits).item()
        next_char = ix2c[next_id]
        generated_text += next_char
        # prepare next input (single token)
        cur_input = torch.tensor([[next_id]], dtype=torch.long)

print("\n--- Generated text (prompt + generated) ---\n")
print(generated_text)
print("\n--- End of generated text ---")
