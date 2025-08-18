import uuid
import pandas as pd
from langchain_chroma import Chroma
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableConfig

df = pd.read_csv('../data/data_topic.csv', encoding='utf-8')

texts = df['text'].tolist()
topics = df['topic'].tolist()

docs = []
for i in range(len(texts)):
    text = texts[i]
    topic = topics[i]
    doc = Document(page_content=text, metadata={'topic': topic})
    docs.append(doc)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

retriever = Chroma(
    persist_directory='./chromaDB_topic',
    embedding_function=embedding_model
).as_retriever(search_kargs={'k': 3})

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

llm = HuggingFacePipeline(pipeline=text_gen)

chats_by_session_id = {}
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

def ask_question(state, config: RunnableConfig):
    session_id = config['configurable']['session_id']
    chat_history = get_chat_history(session_id)

    question = input('질문을 입력해주세요: ').strip()
    topic = input('관련 토픽을 입력해주세요 (공정운영, 유지보수, 이상대응, 전산관리, 안전규정): ').strip()

    retriever = Chroma(
        persist_directory='./chromaDB_topic',
        embedding_function=embedding_model
    ).as_retriever(search_kwargs={'k': 3, 'filter': {'topic': topic}})

    docs = retriever.invoke(question)
    top_docs = docs[:3]
    best_contexts = [doc.page_content for doc in top_docs]
    combined_context = '\n'.join(best_contexts)

    history_text = get_buffer_string(chat_history.messages)
    full_context = f'{history_text}\n\n{combined_context}' if history_text else combined_context

    print(f'question: {question}, context: {full_context}')
    result = text_gen(question=question, context=full_context)

    chat_history.add_user_message(question)
    chat_history.add_ai_message(result['answer'])

    return {
        'question': question,
        'answer': result['answer'],
        'context': best_contexts
    }

def get_answer(state):
    print(f'\n질문: {state["question"]}')
    print(f'답변: {state["answer"]}')
    return state

def ask_reference(state):
    user_input = input('\n참고문서를 보시겠습니까? (예/아니오): ').strip()
    state['show_reference'] = user_input.startswith('예')
    return state

def get_reference(state):
    if state.get('show_reference'):
        print('\n[참고 문서]')
        for i, ctx in enumerate(state['context'], 1):
            print(f'[{i}] {ctx}\n')
    return state

def ask_continue(state):
    user_input = input('\n계속하시겠습니까? (예/아니오): ').strip()
    state['continue'] = user_input.startswith('예')
    return state

def should_show_reference(state):
    return 'get_reference' if state.get('show_reference') else 'ask_continue'

def should_continue(state):
    return 'ask_question' if state.get('continue') else END

graph = StateGraph(dict)

graph.add_node('ask_question', ask_question)
graph.add_node('get_answer', get_answer)
graph.add_node('ask_reference', ask_reference)
graph.add_node('get_reference', get_reference)
graph.add_node('ask_continue', ask_continue)

graph.set_entry_point('ask_question')

graph.add_edge('ask_question', 'get_answer')
graph.add_edge('get_answer', 'ask_reference')
graph.add_conditional_edges('ask_reference', should_show_reference, {
    'get_reference': 'get_reference',
    'ask_continue': 'ask_continue'
})
graph.add_edge('get_reference', 'ask_continue')
graph.add_conditional_edges('ask_continue', should_continue, {
    'ask_question': 'ask_question',
    END: END
})

session_id = str(uuid.uuid4())
config = {'configurable': {'session_id': session_id}}

app = graph.compile()
app.invoke({}, config=config)