import chromadb
import pandas as pd

from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path='./data')

collection = client.get_or_create_collection('test_data01')

collection.add(
    ids = ['doc1', 'doc2'],
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ],
    metadatas = [
        {'text': '안녕하세요'},
        {'text': '날씨가 좋네요'}
    ]
)

query = [[0.1, 0.2, 0.3]]

result = collection.query(query_embeddings=query, n_results=1, include=['embeddings', 'metadatas'])
print(result)

result_text = result['metadatas'][0][0]['text']
print(result_text)

all_data = collection.get()
print(all_data)

doc1_data = collection.get(ids=['doc1'])
print(doc1_data)

collection.add(
    ids = ['doc3'],
    embeddings = [
        [0.7, 0.8, 0.9]
    ],
    metadatas = [
        {'text': '안녕히 가세요'}
    ]
)

all_data = collection.get()
print(all_data)

collection.delete(ids=['doc3'])

all_data = collection.get(include=['embeddings', 'metadatas'])
print(all_data)
df = pd.DataFrame({
    'ids': all_data['ids'],
    'embeddings': [embedding for embedding in all_data['embeddings']],
    'text': [meta['text'] for meta in all_data['metadatas']]
})
print(df.head())

df.to_parquet('./data/test_data01.parquet', index=False)
df2 = pd.read_parquet('./data/test_data01.parquet')
print(df2.head())

df = pd.read_csv('./test_file.csv', encoding='utf8')

sc_list = df['sentence'].tolist()
print(sc_list)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

ids_list, ebd_list, text_list = [], [], []
ith = 0
for sc in sc_list:
    idx = f'doc{ith+1}'
    ids_list.append(idx)
    ith += 1

    text_dic = {'text': sc}
    text_list.append(text_dic)

    embedding = model.encode(sc).tolist()
    ebd_list.append(embedding)

print(ids_list)
print(ebd_list)