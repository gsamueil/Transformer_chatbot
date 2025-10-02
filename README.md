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

