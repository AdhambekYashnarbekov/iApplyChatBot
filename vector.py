from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load your university data
df = pd.read_csv("train.csv", header=None, names=["Question", "Answer"])

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define where to store the Chroma vector DB
db_location = "./chroma_university_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Question"] + " " + row["Answer"],
            metadata={"source": "university_faq"},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
# Create a new Chroma collection for university consulting
vector_store = Chroma(
    collection_name="university_faqs",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
