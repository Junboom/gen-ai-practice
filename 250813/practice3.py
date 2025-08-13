from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import torch

model_name = 'monologg/koelectra-base-v3-finetuned-korquad'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

documents = [
    '우리 회사는 인공지능 연구를 하고 있습니다.',
    '우리 제품은 2023년에 출시되었습니다.'
]
question = '우리 회사는 무슨 연구를 하나요?'

def select_best_context(documents, question):
    result = []
    for doc in documents:
        doc_bool = []
        for word in question.split():
            doc_bool.append(word in doc)
        doc_score = sum(doc_bool)
        result.append(doc_score)
    best_context_idx = result.index(max(result))
    best_context = documents[best_context_idx]
    return best_context

best_context = select_best_context(documents, question)

inputs = tokenizer(question, best_context, return_tensors='pt')
print(inputs)

with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1
print(start_index)
print(end_index)

answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index], skip_special_tokens=True)

print(f'선택된 문서: {best_context}')
print(f'질문: {question}')
print(f'정답: {answer}')