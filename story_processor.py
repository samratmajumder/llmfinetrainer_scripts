import os
import json
from docx import Document
from pptx import Presentation
from bs4 import BeautifulSoup
import spacy

# Load the spaCy NLP model for sentence splitting
nlp = spacy.load("en_core_web_sm")

def extract_text_from_docx(file_path):
    """Extract text from a .docx file."""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def extract_text_from_pptx(file_path):
    """Extract text from a .pptx file."""
    try:
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text += shape.text.strip() + "\n"
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def extract_text_from_html(file_path):
    """Extract text from an .html file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator="\n").strip()
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

# NEW FUNCTION: Extract text from .txt files
def extract_text_from_txt(file_path):
    """Extract text from a .txt file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def split_into_sentences(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def create_prompt_completion(text, num_sentences=3):
    """Create a prompt-completion pair from the text."""
    if not text:
        return None, None
    sentences = split_into_sentences(text)
    if not sentences:
        return None, None
    # Use the first 'num_sentences' as prompt, or all if fewer exist
    prompt_sentences = sentences[:min(num_sentences, len(sentences))]
    prompt = " ".join(prompt_sentences)
    completion = text  # Entire story as completion
    return prompt, completion

def process_folder(folder_path, output_jsonl, num_sentences=3):
    """Process all supported files in the folder and create a JSONL file."""
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name.lower())
            # MODIFIED: Added .txt to supported extensions
            if file_name.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            elif file_name.endswith(".pptx"):
                text = extract_text_from_pptx(file_path)
            elif file_name.endswith(".html"):
                text = extract_text_from_html(file_path)
            elif file_name.endswith(".txt"):  # NEW: Handle .txt files
                text = extract_text_from_txt(file_path)
            else:
                print(f"Skipping unsupported file: {file_name}")
                continue

            if text:
                prompt, completion = create_prompt_completion(text, num_sentences)
                if prompt and completion:
                    json_obj = {"prompt": prompt, "completion": completion}
                    f.write(json.dumps(json_obj) + "\n")
                    print(f"Processed: {file_name}")
            else:
                print(f"No text extracted from: {file_name}")

if __name__ == "__main__":
    # Example usage
    folder_path = "path/to/your/stories"  # Replace with your folder path
    output_jsonl = "training_data.jsonl"   # Output file name
    num_sentences = 3                      # Number of sentences for prompt

    print(f"Starting processing of folder: {folder_path}")
    process_folder(folder_path, output_jsonl, num_sentences)
    print(f"Training data saved to: {output_jsonl}")