import sys

file_path = r'src/train.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    '⚠': 'WARNING:',
    '≥': '>=',
    '≤': '<=',
    '→': '->',
    '—': '-',
    '─': '-',
    '…': '...',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'"
}

for k, v in replacements.items():
    text = text.replace(k, v)

remaining = 0
chars = list(text)
for i, c in enumerate(chars):
    if ord(c) > 127:
        remaining += 1
        chars[i] = ' '
text = "".join(chars)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Fix complete. Remaining unhandled non-ASCII replaced with space: {remaining}')
