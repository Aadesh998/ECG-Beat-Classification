import os

# Define directories and file types for detailed content extraction
INCLUDE_DIRS = [
    "client",
    "server",
]
INCLUDE_EXTENSIONS = [
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".css",
    ".html"
]

EXCLUDE_DIRS = ["node_modules", ".git", "public", "build", ".next", "venv", "__pycache__"]

EXCLUDE_FILES = [
    "package-lock.json",
    "package.json",
    ".gitignore",
    ".env",
    "requirements.txt",
    "README.md"
    "poetry.lock",
    "localrtt_microservices.py",
    "microservices-app-code-files.txt",
    "yarn.lock",
    "tsconfig.json",
    "*.json",
    "next.config.mjs",
    "next-env.d.ts",
    ".eslintrc.json",
    "postcss.config.mjs",
    "tailiwind.config.json",
    "system_documentation.md",  # Add any files you want to exclude (if needed)
]


def get_file_folder_structure(repo_path):
    """
    Generate a concise representation of the file/folder structure.
    """
    structure = "File/Folder Structure:\n"
    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        level = root.replace(repo_path, "").count(os.sep)
        indent = " " * 4 * level
        structure += f"{indent}{os.path.basename(root)}/\n"
        subindent = " " * 4 * (level + 1)
        for f in files:
            if f not in EXCLUDE_FILES and any(
                f.endswith(ext) for ext in INCLUDE_EXTENSIONS
            ):
                structure += f"{subindent}{f}\n"
    return structure + "\n"


def extract_code_files(repo_path):
    """
    Extract and compile contents from code files, adding headers for each file.
    """
    compiled_contents = ""

    # Append the file/folder structure
    compiled_contents += get_file_folder_structure(repo_path)

    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file in EXCLUDE_FILES:
                continue
            if any(file.endswith(ext) for ext in INCLUDE_EXTENSIONS):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        compiled_contents += "---\n"
                        compiled_contents += (
                            f"File: {os.path.relpath(file_path, start=repo_path)}\n\n"
                        )
                        compiled_contents += "```"
                        if file.endswith(".py"):
                            compiled_contents += "python"
                        elif file.endswith((".ts", ".tsx")):
                            compiled_contents += "typescript"
                        elif file.endswith((".js", ".jsx")):
                            compiled_contents += "javascript"
                        elif file.endswith(".yml"):
                            compiled_contents += "yaml"
                        elif file.endswith((".css", ".scss")):
                            compiled_contents += "css"
                        elif file.endswith(".md"):
                            compiled_contents += "markdown"
                        compiled_contents += "\n"
                        compiled_contents += f.read() + "\n"
                        compiled_contents += "```\n\n"
                except UnicodeDecodeError:
                    compiled_contents += "---\n"
                    compiled_contents += (
                        f"File: {os.path.relpath(file_path, start=repo_path)}\n"
                    )
                    compiled_contents += (
                        "Content: Skipped due to unsupported encoding\n\n"
                    )
    return compiled_contents


if __name__ == "__main__":
    repo_path = input("Please enter the path to the local repository: ")
    code_files_content = extract_code_files(repo_path)
    output_filename = "gmail-mcp-codebase.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(code_files_content)
    print(f"Code files content compiled to '{output_filename}'.")
