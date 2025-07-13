import os

search_term = "support"
root_dir = "."  # Current directory; change if needed

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(subdir, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if search_term in line:
                        print(f"{filepath} (line {i}): {line.strip()}")
