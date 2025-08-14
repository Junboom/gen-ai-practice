import pandas as pd

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate

df = pd.read_csv('../data/data.csv', encoding='utf-8')

texts = df['text'].tolist()
docs = [Document(page_content=text) for text in texts]
print(docs)

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

vertorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory='./chromaDB1'
)

retriever = Chroma(
    persist_directory='./chromaDB1',
    embedding_function=embedding_model
).as_retriever(search_kwarg={'k': 3})

question = '클린룸 입장 시 주의사항 알려줄래?'
chroma_docs = retriever.invoke(question)
best_context = chroma_docs[0].page_content
print('참고 문서: ', best_context)
print(chroma_docs)

model_id = 'monologg/koelectra-base-v3-finetuned-korquad'
qa_tokenizer = AutoTokenizer.from_pretrained(model_id)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_id)

text_gen = pipeline(
    'question-answering',
    model=qa_model,
    tokenizer=qa_tokenizer,
    device=-1,
    max_length=512,
    do_sample=False,
    temperature=0.1,
    truncation=True
)

result = text_gen(question=question, context=best_context)
print('정답: ', result['answer'])

gen_model_id = 'EleutherAI/polyglot-ko-1.3b'
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_id)

gen_pipeline = pipeline(
    'text-generation',
    model=gen_model,
    tokenizer=gen_tokenizer,
    device=-1,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    truncation=True
)

gen_llm = HuggingFacePipeline(pipeline=gen_pipeline)

prompt = PromptTemplate.from_template("""
너는 똑똑한 제조 전문가야. 아래의 질문과 정답을 바탕으로 자연스럽고 친절하게 응답해줘.

질문: {question}
정답: {answer}

자연어 응답:
""")

chain = prompt | gen_llm
final_response = chain.invoke({
    'question': question,
    'answer': result['answer']
})

print('\n 자연어 응답:')
print(final_response)