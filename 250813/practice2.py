import numpy as np

documents = [
    '파이썬은 범용 프로그래밍 언어입니다.',
    '서울은 대한민국의 수도입니다.',
    'RAG는 검색과 생성을 결합한 기술입니다.',
    'GPT는 자연어 생성을 위한 딥러닝 모델입니다.',
    '축구는 세계에서 가장 인기 있는 스포츠입니다.'
]
query = 'RAG는 무엇인가요?'

def tokenizer(text):
    res = []
    text_split = text.replace('?', '').replace('.', '').split()
    for word in text_split:
        res.append(word.lower())
    return res

def count_vector(tokens, voca):
    res = []
    for word in voca:
        res.append(tokens.count(word))
    return res

def cosine_similarity(vec1, vec2):
    res = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return res

tokenized_docs = [tokenizer(doc) for doc in documents]
query_tokens = tokenizer(query)

print(tokenized_docs)
print(query_tokens)

voca = sorted(set(query_tokens + sum(tokenized_docs, [])))

query_vec = count_vector(query_tokens, voca)
similarities = []
for doc_tokens in tokenized_docs:
    doc_vec = count_vector(doc_tokens, voca)
    sim = cosine_similarity(query_vec, doc_vec)
    similarities.append(sim)

def generate_answer(query, context):
    res = f'"{context}" 라는 내용을 보면 "{query}"에 대한 답을 유추할 수 있어요.'
    return res

retrieved_doc = documents[np.argmax(similarities)]

print("질문: ", query)
print("관련 문서: ", retrieved_doc)
print("생성된 답변: ", generate_answer(query, retrieved_doc))