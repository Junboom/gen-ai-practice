import pandas as pd

from langchain_chroma import Chroma
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

df = pd.read_csv('../data/data.csv', encoding='utf-8')
print(df.head(10))

texts = df['text'].tolist()
docs = [Document(page_content=text) for text in texts]

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory='./chromaDB'
)

model_id = 'skt/kogpt2-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

text_gen = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    truncation=False,
    max_new_tokens=100,
    do_sample=True,
    return_full_text=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=text_gen)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

def ask_question(state):
    question = input('질문을 입력해주세요: ')
    result = qa_chain.invoke({'query': question})
    res = {
        'question': question,
        'answer': result['result'],
        'source_docs': result['source_documents']
    }
    return res

def get_answer(state):
    print('\n답변: ', state['answer'])
    return state

def ask_reference(state):
    reply = input('\n참고문서를 보시겠습니까? (예/아니오): ').strip()
    return {**state, 'see_reference': reply}

def get_reference(state):
    print('\n[참고 문서]')
    for i, doc in enumerate(state['source_docs'], 1):
        print(f'{i}. {doc.page_content[:200]}...')
    return state

def ask_continue(state):
    reply = input('\n계속하시겠습니까? (예/아니오): ').strip()
    return {**state, 'continue': reply}

def reference_or_not(state):
    return 'get_reference' if state.get('see_reference') == '예' else 'ask_continue'

def continue_or_not(state):
    return 'ask_question' if state.get('continue') == '예' else END

graph = StateGraph(dict)

graph.add_node('ask_question', ask_question)
graph.add_node('get_answer', get_answer)
graph.add_node('ask_reference', ask_reference)
graph.add_node('get_reference', get_reference)
graph.add_node('ask_continue', ask_continue)

graph.set_entry_point('ask_question')

graph.add_edge('ask_question', 'get_answer')
graph.add_edge('get_answer', 'ask_reference')
graph.add_conditional_edges('ask_reference', reference_or_not, {
    'get_reference': 'get_reference',
    'ask_continue': 'ask_continue'
})
graph.add_edge('get_reference', 'ask_continue')
graph.add_conditional_edges('ask_continue', continue_or_not, {
    'ask_question': 'ask_question',
    END: END
})

app = graph.compile()
app.invoke({})