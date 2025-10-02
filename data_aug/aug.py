import os
import json
import requests
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator="\n")
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


def load_data(input_data, is_url=False):
    if is_url:
        return read_url(input_data)

    ext = os.path.splitext(input_data)[-1].lower()
    if ext == ".txt":
        return read_txt(input_data)
    elif ext == ".docx":
        return read_docx(input_data)
    elif ext == ".pdf":
        return read_pdf(input_data)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def save_for_training(text, output_file="training_data.jsonl", chunk_size=300):

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            entry = {
                "instruction": "Answer questions based on company/business data.",
                "input": "",
                "output": chunk.strip()
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks into {output_file}")

if __name__ == "__main__":

    text_data = load_data("random.pdf") # LINK UR PDF PATH HERE
    save_for_training(text_data, "business_data.jsonl")


    url_text = load_data("https:random.url", is_url=True) # LINK UR LINK UP HERE
    save_for_training(url_text, "website_data.jsonl")
