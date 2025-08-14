from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def classification(state):
    text = state.get('user_input', '')

    if any(word in text for word in ['좋아요', '네', '좋다', '응']):
        label = 'positive'
    elif any(word in text for word in ['싫어요', '아니요', '별로', '싫다']):
        label = 'negative'
    else:
        label = 'neutral'
    return {**state, 'label': label}

def positive_answer(state):
    return {**state, 'response': '좋게 생각해주셔서 감사합니다!'}

def negative_answer(state):
    return {**state, 'response': '무엇이 불편하셨나요?'}

def neutral_answer(state):
    return {**state, 'response': '조금 더 자세히 말씀해 주세요.'}

graph = StateGraph(dict)

graph.add_node('classification', classification)
graph.add_node('positive', positive_answer)
graph.add_node('negative', negative_answer)
graph.add_node('neutral', neutral_answer)

graph.set_entry_point('classification')

def get_label(state):
    return state.get('label', '')

graph.add_conditional_edges('classification', get_label, {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral'
})

graph.add_edge('positive', END)
graph.add_edge('negative', END)
graph.add_edge('neutral', END)

app = graph.compile()

user_input = input('한글 입력: ')

final_state = app.invoke({'user_input': user_input})
print('응답: ', final_state.get('response', ''))

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

prompt = PromptTemplate.from_template('아주 먼 옛날 {user_input}이 살고 있었습니다. 그러던 어느날 ')
chain = prompt | llm

result1 = chain.invoke({'user_input': '토끼'})
print(result1)
print(len(result1))

result2 = chain.invoke({'user_input': '고양이'})
print(result2)
print(len(result2))

def classification(state):
    user_input = state.get('user_input', '')
    result = chain.invoke({'user_input': user_input}).strip()

    if len(result) > 250:
        label = 'long_answer'
    else:
        label = 'short_answer'

    return {**state, 'label': label}

def long_answer(state):
    return {**state, 'response': '장편 소설이 생성되었습니다.'}

def short_answer(state):
    return {**state, 'response': '단편 소설이 생성되었습니다.'}

def get_label(state):
    return state.get('label', '')

graph = StateGraph(dict)

graph.add_node('classification', classification)
graph.add_node('long_answer', long_answer)
graph.add_node('short_answer', short_answer)

graph.set_entry_point('classification')

graph.add_conditional_edges('classification', get_label, {
    'long_answer': 'long_answer',
    'short_andwer': 'short_answer'
})

graph.add_edge('long_answer', END)
graph.add_edge('short_answer', END)

app = graph.compile()

user_input = input('이야기를 만들고 싶은 동물 입력: ')
final_state = app.invoke({'user_input': user_input})
print('응답: ', final_state.get('response', ''))