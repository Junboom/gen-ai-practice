from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain_huggingface import HuggingFacePipeline

model_id = 'monologg/koelectra-base-v3-finetuned-korquad'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_id)

qa_pipeline = pipeline(
    'question-answering',
    model=model,
    tokenizer=tokenizer,
    device=-1
)

question = '세종대왕은 어떤 업적을 남겼나요?'
context = '세종대왕은 조선의 네 번쨰 왕으로, 한글을 창제하고 과학, 음악, 농업 등 다양한 분야에서 업적을 남겼습니다. 그는 집현전을 설치하고 학문을 장려하였으며, 측우기와 해시계 등의 과학 기구 개발을 지원했습니다.'

response = qa_pipeline(question=question, context=context)
print('답변: ', response['answer'])