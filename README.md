# Transformer Chatbot 🤖✨
A lightweight Transformer-based chatbot project with training utilities and a small Flask web UI.  
This repo contains preprocessing, tokenizer utilities, a training script, a small dataset, and a Flask app for quick local demo/deployment.

---

## Why this project?
- Build and experiment with a simple Transformer-based conversational model.
- Includes data preprocessing and tokenizer utilities to run experiments quickly.
- Provides a minimal web UI (Flask) so you can interact with the trained model instantly.
- Great starting point for classroom demos, quick POCs, and iterative research.

---

## Table of contents
- [Quickstart](#quickstart)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Preprocess dataset](#preprocess-dataset)
- [Training](#training)
- [Serve / Demo (Flask)](#serve--demo-flask)
- [Files of interest](#files-of-interest)
- [Tips & Tricks](#tips--tricks)
- [Roadmap & Ideas](#roadmap--ideas)
- [Contributing](#contributing)
- [License](#license)

---

## Quickstart
1. Clone the repo:
   ```bash
   git clone https://github.com/gsamueil/Transformer_chatbot.git
   cd Transformer_chatbot
Create and activate a virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


Install dependencies:

pip install torch transformers flask sentencepiece nltk tqdm


Or use pip install -r requirements.txt if you create one.

Preprocess data and train (examples below).

Requirements

Python 3.8+

PyTorch

Hugging Face transformers

Flask

nltk, sentencepiece

tqdm, numpy, pandas (optional)
Install with:

pip install torch transformers flask nltk sentencepiece tqdm numpy pandas

Repository structure
Transformer_chatbot/
├── data.txt              # dataset
├── preprocess.py         # preprocessing pipeline
├── my_tokenizer.py       # tokenizer utilities
├── train_qabot_words.py  # training script
├── chat_words.py         # inference script
├── flask_app.py          # Flask demo
├── templates/            # HTML templates
└── static/               # static assets (CSS/JS)

Preprocess the dataset
python preprocess.py --input data.txt --output processed.jsonl


(Replace flags with actual ones used in your script.)

Training
python train_qabot_words.py \
  --data processed.jsonl \
  --model_dir models/qabot-v1 \
  --epochs 5 \
  --batch_size 16 \
  --lr 5e-5

Serve / Demo (Flask)

Start the Flask app:

python flask_app.py


Visit http://127.0.0.1:5000/
 in your browser.
API example:

curl -X POST http://127.0.0.1:5000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"Hello, how are you?"}'

Using the tokenizer
from my_tokenizer import MyTokenizer
tok = MyTokenizer()
tok.train("processed.txt", vocab_size=8000)
tok.save("tokenizer.model")
tok = MyTokenizer.load("tokenizer.model")
print(tok.encode("Hello!"))

Example workflow

Preprocess:

python preprocess.py --input data.txt --output processed.jsonl


Train tokenizer:

python my_tokenizer.py --train processed.jsonl --out tokenizer.model


Train model:

python train_qabot_words.py --data processed.jsonl --tokenizer tokenizer.model --model_dir models/qabot-v1


Run demo:

python flask_app.py --model models/qabot-v1 --tokenizer tokenizer.model

Files of interest

flask_app.py — web UI & REST API

train_qabot_words.py — training loop

preprocess.py — dataset preprocessing

my_tokenizer.py — tokenizer utilities

chat_words.py — inference wrapper

Troubleshooting

Flask errors → check console logs, confirm model path.

GPU not used → torch.cuda.is_available()

Tokenization mismatch → ensure same tokenizer used at train + inference.

Memory issues → reduce batch_size.

Roadmap & Ideas

Add requirements.txt

Add Dockerfile

Provide Colab notebook demo

Add unit tests for tokenizer and preprocess

Implement evaluation (BLEU, perplexity)

Contributing

Fork the repo

Create branch: git checkout -b feat/your-feature

Commit + push: git push origin feat/your-feature

Open PR

License

license (MIT
