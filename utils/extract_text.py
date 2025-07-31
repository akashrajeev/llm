import os
import fitz  # PyMuPDF
import docx
import extract_msg
import email
import json
from email import policy

# -------------------------------
# PDF (.pdf) Extraction
# -------------------------------
def extract_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
    return text

# -------------------------------
# Word Document (.docx) Extraction
# -------------------------------
def extract_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Error reading DOCX: {e}")
    return text

# -------------------------------
# Outlook Email (.msg) Extraction
# -------------------------------
def extract_from_msg(file_path):
    try:
        msg = extract_msg.Message(file_path)
        subject = msg.subject or ""
        body = msg.body or ""
        return f"Subject: {subject}\n\n{body}"
    except Exception as e:
        print(f"❌ Error reading MSG: {e}")
        return ""

# -------------------------------
# Raw Email (.eml) Extraction
# -------------------------------
def extract_from_eml(file_path):
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        subject = msg["subject"] or ""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content()
        else:
            body = msg.get_content()
        return f"Subject: {subject}\n\n{body}"
    except Exception as e:
        print(f"❌ Error reading EML: {e}")
        return ""

# -------------------------------
# File Router
# -------------------------------
def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext == ".docx":
        return extract_from_docx(file_path)
    elif ext == ".msg":
        return extract_from_msg(file_path)
    elif ext == ".eml":
        return extract_from_eml(file_path)
    else:
        raise ValueError("❌ Unsupported file type: " + ext)

# -------------------------------
# Optional: Chunk Text (for LLM input)
# -------------------------------
def chunk_text(text, max_length=500):
    chunks = []
    paragraph = ""
    for line in text.split("\n"):
        if len(paragraph) + len(line) < max_length:
            paragraph += line + " "
        else:
            chunks.append(paragraph.strip())
            paragraph = line + " "
    if paragraph:
        chunks.append(paragraph.strip())
    return chunks

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    test_file = r"C:\Users\Akash\Downloads\BAJHLIP23020V012223.pdf"
    
    full_text = extract_text(test_file)
    chunks = chunk_text(full_text)

    # Save full extracted text
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # Save chunked text for embeddings
    with open("chunked_text.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Print summary and sample chunks
    print(f"\n✅ Extracted {len(chunks)} text chunks (ready for embedding)\n")
    print("Saved to:")
    print(" - extracted_text.txt (raw full text)")
    print(" - chunked_text.json (text chunks)")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:\n{chunk}\n---")
