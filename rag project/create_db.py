import os
import shutil
import tkinter as tk
from tkinter import filedialog

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---------- NEW: open a file picker instead of typing the path ----------
root = tk.Tk()
root.withdraw()                      # hide the empty main window
root.attributes("-topmost", True)    # make the dialog appear in front

pdf_paths = filedialog.askopenfilenames(
    title="Select one or more PDF files",
    filetypes=[("PDF files", "*.pdf")]
)
root.destroy()

if not pdf_paths:
    print("No file selected. Exiting.")
    raise SystemExit

if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
# -------------------------------------------------------------------------

docs = []
for path in pdf_paths:
    docs.extend(PyPDFLoader(path).load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=30
)
chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

print(f"Database created from {len(pdf_paths)} PDF(s):")
for p in pdf_paths:
    print(" -", os.path.basename(p))