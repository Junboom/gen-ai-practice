from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('./data/doc.txt', encoding='utf-8') as f:
    raw_text = f.read()

print(raw_text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=['\n', '.', ' ']
)
chunks = text_splitter.create_documents([raw_text])

embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory='./chromaDB_chunked'
)

vectorstore = Chroma(
    persist_directory='./chromaDB_chunked',
    embedding_function=embedding_model
)

collection = vectorstore._collection
docs = collection.get()
print(docs)

docs = vectorstore._collection.get(limit=5)

for i in range(len(docs['documents'])):
    print(f'Document {i+1}: {docs["documents"][i]}')

docs = vectorstore._collection.get(limit=3, include=['embeddings', 'documents'])

for i, embedding in enumerate(docs['embeddings']):
    print(f'\n문서 {i+1} 임베딜 벡터 (길이: {len(embedding)})')
    print(embedding[:10], '...')