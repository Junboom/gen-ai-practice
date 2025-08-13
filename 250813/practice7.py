import pandas as pd
import torch
import chromadb

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

df = pd.read_csv('./data/data.csv', encoding='utf8')
print(df.head(10))

sc_list = df['text'].tolist()
print(sc_list)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

ids_list, ebd_list, text_list = [], [], []
for i, sc in enumerate(sc_list):
    doc_id = f'doc{i+1}'
    ids_list.append(doc_id)
    text_list.append({'text': sc})

    embedding = model.encode(sc).tolist()
    ebd_list.append(embedding)

client = chromadb.PersistentClient(path='./chromaDB')
collection = client.get_or_create_collection('company_doc')

collection.add(
    ids=ids_list,
    embeddings=ebd_list,
    metadatas=text_list
)

question = '야간조 작업자 작업 전'
question_embedding = [model.encode(question).tolist()]
# print(question_embedding)

result = collection.query(query_embeddings=question_embedding, n_results=1)
retrieved_text = result['metadatas'][0][0]['text']
print(f'선택된 문서: {retrieved_text}')

qa_model_name = 'monologg/koelectra-base-v3-finetuned-korquad'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

inputs = qa_tokenizer(question, retrieved_text, return_tensors='pt', truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = qa_model(**inputs)

start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1

answer = qa_tokenizer.decode(inputs['input_ids'][0][start_index:end_index], skip_special_tokens=True)

print(f'질문: {question}')
print(f'정답: {answer}')