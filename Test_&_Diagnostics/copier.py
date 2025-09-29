import os

# Gather file extensions you want to include
extensions = ["py", "txt", "c", "s", "o", "cpp", "h", "java", "js", "sh", "md", "cu"]
folder = os.path.abspath("./")  # or your desired folder
folder_name = os.path.basename(folder.rstrip("/"))

output_file = os.path.join(folder, f"{folder_name}.txt")

with open(output_file, "w", encoding="utf-8", errors="replace") as out:
    for filename in os.listdir(folder):
        if any(filename.endswith(f".{ext}") for ext in extensions):
            out.write(f"\n# --- {filename} ---\n\n")
            with open(os.path.join(folder, filename), "r", encoding="utf-8", errors="replace") as f:
                out.write(f.read())
print(f"\nDonksie Bo Bunskie:{output_file}")

