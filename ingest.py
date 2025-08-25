from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load your PDF
loader = PyPDFLoader("data/sample.pdf")  # make sure your PDF is named sample.pdf
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# Create embeddings & save in vector DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(splits, embedding=embeddings, persist_directory="storage")
db.persist()

print("✅ Document ingested and stored successfully!")
