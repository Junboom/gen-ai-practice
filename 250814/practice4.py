import pandas as pd

from sentence_transformers import SentenceTransformer, util
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering, pipeline
from langchain_core.prompts import PromptTemplate

model_id = 'skt/kogpt2-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

text_gen = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=-1,
    truncation=True,
    max_length=256,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=text_gen)
prompt = PromptTemplate.from_template('산 속에 {animal} 한 마리가 살고 있었습니다. 그러던 어느 날 ')
chain = prompt | llm

response = chain.invoke({'animal': '토끼'})
print(response)

df = pd.read_csv('../data/data.csv', encoding='utf-8')
print(df.head(10))

documents = df['text'].tolist()

embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

qa_model_name = 'monologg/koelectra-base-v3-finetuned-korquad'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

text_gen = pipeline(
    'question-answering',
    model=qa_model,
    tokenizer=qa_tokenizer,
    device=-1,
    truncation=True,
    max_length=512,
    do_sample=True,
    temperature=0.1
)

question = '클린룸 입장 시 주의사항 알려줄래?'
question_embedding = embedding_model.encode(question, convert_to_tensor=True)
cos_sim = util.cos_sim(question_embedding, doc_embeddings)[0]
best_idx = cos_sim.argmax().item()
best_context = documents[best_idx]
print('참고 문서: ', best_context)

qa_input = {
    'question': question,
    'context': best_context
}

result = text_gen(question=question, context=best_context)
print('정답: ', result['answer'])