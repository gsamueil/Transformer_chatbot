# preprocess.py
with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

processed_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith("Q:"):
        line = line.replace("Q:", "<q>") + " <eos>"
    elif line.startswith("A:"):
        line = line.replace("A:", "<a>") + " <eos>"
    processed_lines.append(line)

with open("processed_data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(processed_lines))

print("✅ Data processed and saved into processed_data.txt")
