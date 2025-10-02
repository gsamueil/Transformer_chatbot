from flask import Flask, render_template, request, jsonify
import torch
import joblib
from train_qabot import TransformerLM, device

app = Flask(__name__)

# تحميل الـ vocab
stoi = joblib.load("stoi.pkl")
itos = joblib.load("itos.pkl")

# تعريف encode / decode
def encode(text):
    tokens = text.split()
    return [stoi[t] for t in tokens if t in stoi]

decode = lambda ids: " ".join([itos.get(i, "<UNK>") for i in ids])

# تحميل الموديل
model = TransformerLM(len(stoi)).to(device)
model.load_state_dict(torch.load("qabot_words.pt", map_location=device))
model.eval()

# 🔹 دالة الإجابة (بدل ask)
def answer_question(question, max_new_tokens=100, temperature=0.8, top_k=50):
    """
    - بناء الـ prompt: <q> question <a>
    - يولد حتى <eos>, و يرجع نص الإجابة (دون الـ prompt أو <eos>)
    """
    prompt = f"<q> {question.strip()} <a>"
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token=stoi["<eos>"]
    )

    out_ids = out[0].tolist()
    gen_ids = out_ids[len(input_ids[0]):]

    # قطع عند أول <eos> لو موجود
    if stoi["<eos>"] in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(stoi["<eos>"])]

    # decode
    answer = decode(gen_ids).strip()
    return answer if answer else "⚠️ لم أستطع توليد إجابة مناسبة."

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    response = answer_question(user_msg)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
