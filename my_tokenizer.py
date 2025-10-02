# tokenize.py
import re

def tokenize(text):
    text = text.lower()
    # Includes words + numbers + currency symbols + special marks
    tokens = re.findall(r"[a-zA-Z0-9]+|[\$€£¥%&.,!?;:()\"'-]", text)
    return tokens
