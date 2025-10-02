import math, torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from my_tokenizer import tokenize  # ✅ التوكنيزر بتاعك

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# 1) Data & Tokenization
# -----------------------
with open("processed_data.txt", "r", encoding="utf-8") as f:
    text = f.read()  # ملفنا بالفعل lower() من preprocess
# توكنيز
tokens = tokenize(text)

# الآن يجب أن ترى <q>, <a>, <eos> ضمن tokens
vocab = sorted(set(tokens))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}

# safety: تأكد أن '<eos>' موجود (لو preprocess صحيح فـ موجود)
# assert "<eos>" in stoi, "ERROR: <eos> not found in vocab — run preprocess.py and re-train."

encode = lambda s: [stoi[w] for w in tokenize(s) if w in stoi]
decode = lambda ids: " ".join([itos[i] for i in ids])

# تحويل النص إلى indices
data = torch.tensor([stoi[w] for w in tokens], dtype=torch.long)
n = int(len(data) * 0.9)
train_data, val_data = data[:n], data[n:]

# -----------------------
# 2) Hyperparameters
# -----------------------
batch_size = 16
block_size = 64
n_embed = 128
n_head = 4
n_layer = 2
dropout = 0.1
learning_rate = 3e-4
max_iters = 500

# -----------------------
# 3) Batch function
# -----------------------
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

# -----------------------
# 4) Transformer Model
# -----------------------
class Head(nn.Module):
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        wei = q @ k.transpose(-2,-1) / (C**0.5)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([Head(n_embed, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_embed, n_head)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = FeedForward(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T,-1), targets.view(B*T))
        return logits, loss
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, eos_token=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_token is not None and next_id.item() == eos_token:
                break
        return idx
    

# -----------------------
# 5) Training Loop
# -----------------------
model = TransformerLM(len(vocab)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step} loss {loss.item():.4f}")

# -----------------------
# Save model + vocab
# -----------------------
torch.save(model.state_dict(), "qabot_words.pt")
joblib.dump(stoi, "stoi.pkl")
joblib.dump(itos, "itos.pkl")
print("✅ Model and vocab saved: qabot_words.pt, stoi.pkl, itos.pkl")

# -----------------------
# Example inference
# -----------------------
# q = "[Q] what is gpt?"
# input_ids = torch.tensor([encode(q)], dtype=torch.long).to(device)
# output_ids = model.generate(input_ids, max_new_tokens=100, eos_token=stoi["[EOS]"])
# print("🔹", decode(output_ids[0].tolist()))
