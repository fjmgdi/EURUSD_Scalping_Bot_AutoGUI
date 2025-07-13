import os

search_term = 'signal'

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if search_term in line:
                        print(f"{filepath} (line {i+1}): {line.strip()}")
            except Exception as e:
                print(f"Could not read {filepath}: {e}")
