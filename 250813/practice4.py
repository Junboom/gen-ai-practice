import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util

qa_model_name = 'monologg/koelectra-base-v3-finetuned-korquad'
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print(embedding_model)

df = pd.read_csv("test_file.csv")
print(df)

documents = [
    '우리 회사는 인공지능 연구를 하고 있습니다.',
    '우리 제품은 2023년에 출시되었습니다.'
]
question = '우리 회사는 무슨 연구를 하나요?'

doc_embeddings = embedding_model.encode(documents)
query_embedding = embedding_model.encode(question)

cosine_similarity = util.cos_sim(query_embedding, doc_embeddings)[0]
print(cosine_similarity)

best_idx = cosine_similarity.argmax().item()
print(best_idx)

best_context = documents[best_idx]
print(best_context)

inputs = tokenizer(question, best_context, return_tensors='pt')

with torch.no_grad():
    outputs = qa_model(**inputs)

start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1
print(start_index)
print(end_index)

answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index], skip_special_tokens=True)

print(f'선택된 문서: {best_context}')
print(f'질문: {question}')
print(f'정답: {answer}')