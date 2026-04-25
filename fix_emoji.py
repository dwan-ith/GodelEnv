import sys

with open('README.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('emoji:'):
        # Enforce valid train emoji for HF
        new_lines.append('emoji: "🚞"\n')
    else:
        new_lines.append(line)

with open('README.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
