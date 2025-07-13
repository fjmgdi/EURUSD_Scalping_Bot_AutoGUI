import os

def replace_in_file(filepath, old_text, new_text):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_text in content:
        content_new = content.replace(old_text, new_text)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content_new)
        print(f"Replaced '{old_text}' with '{new_text}' in {filepath}")

def main():
    root_dir = os.getcwd()  # Run script in your project root folder
    old_text = "signal"
    new_text = "signal"

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(subdir, file)
                replace_in_file(filepath, old_text, new_text)

if __name__ == "__main__":
    main()
