import torch, re
import joblib
from train_qabot_words import TransformerLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 vocab 불러오기
stoi = joblib.load("stoi.pkl")
itos = joblib.load("itos.pkl")
vocab_size = len(stoi)

encode = lambda s: [stoi.get(w, 0) for w in s.lower().split()]
decode = lambda ids: " ".join([itos[i] for i in ids])

# 🔹 모델 불러오기
model = TransformerLM(vocab_size).to(device)
model.load_state_dict(torch.load("qabot_words.pt", map_location=device))
model.eval()

print("💬 챗봇 시작! (quit/exit 입력 시 종료)")

# 🔹 대화 루프
while True:
    q = input("너: ")
    if q.strip().lower() in ["quit", "exit"]:
        print("대화 종료!")
        break

    # Q: 질문 → A: 답변 형태로 prompt
    prompt = f"q: {q.lower()} a:"
    x = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    out = model.generate(x, max_new_tokens=40)
    answer = decode(out[0].tolist())

    # ✅ 후처리: prompt 부분 제거 & 다음 Q: 나오기 전에 자르기
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    answer = re.split(r"q\s*:", answer)[0].strip()

    print("봇:", answer)
