
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma vector database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Local Ollama model
llm = ChatOllama(
    model="gemma3:1b"
)

# Prompt template
template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an intelligent AI assistant. Answer user questions strictly using the provided PDF/document context.

Instructions:
1. Understand the entire context before answering.
2. Automatically detect and correct spelling mistakes, typos, grammatical errors, and missing letters in user questions.
3. Understand the user's intended meaning instead of relying on exact keyword matching.
4. Use synonyms, similar words, fuzzy matching, and semantic search to find relevant information in the PDF.
5. Search the entire context for the most relevant answer, even if the question is worded differently.
6. Give accurate, simple, and clear answers based only on the document. Never make up information.
7. Do not reject a question just because of spelling or grammar mistakes.
8. Only if the answer genuinely cannot be found after trying all matching methods, respond exactly:

"I could not find the answer in the document." """),

        ("human", """
        content: {context}
        question: {question}
        """)
    ]
)

print("RAG system created")
print("Press 0 to exit")

# Main loop
while True:
    query = input("You: ")

    if query == "0":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Combine document content
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prepare prompt
    final_prompt = template.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Generate response using Ollama
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
  
    