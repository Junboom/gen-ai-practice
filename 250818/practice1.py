from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

prompt = PromptTemplate.from_template('아주 먼 옛날 {user_input}이 살고 있었습니다. 그러던 어느 날 ')
story_chain = prompt | llm

def classification(state):
    user_input = state.get('user_input', '')
    story = story_chain.invoke({'user_input': user_input}).strip()
    label = 'long_answer' if len(story) > 256 else 'short_answer'
    return {**state, 'label': label, 'story': story}

def long_answer(state):
    res = '장편 소설이 생성되었습니다.'
    print(res)
    return {**state, 'response': res}

def short_answer(state):
    res = '단편 소설이 생성되었습니다.'
    print(res)
    return {**state, 'response': res}

def ask_continue(state):
    reply = input('\n추가 질문을 하시겠습니까? (예/아니오)\n> ').strip()
    if reply in ['예', '네', 'yes', 'Yes']:
        state['continue'] = True
        question = input('추가 질문을 입력하세요:\n> ').strip()
        state['user_input'] = question
    else:
        state['continue'] = False
    return state

def should_continue(state):
    if state.get('continue'):
        answer = 'replay'
    else:
        answer = '__end__'
    return answer

def replay_story(state):
    question = state.get('user_input', '')
    if '보여줘' in question:
        print('\n이전에 생성된 이야기:\n', state.get('story', '[이야기가 없습니다]'))
        return {**state, 'response': '이전 이야기 다시 보여줌'}
    else:
        return classification(state)
    
def get_label(state):
    return state.get('label', '')

graph = StateGraph(dict)
graph.add_node('classification', classification)
graph.add_node('long_answer', long_answer)
graph.add_node('short_answer', short_answer)
graph.add_node('ask_continue', ask_continue)
graph.add_node('replay', replay_story)

graph.set_entry_point('classification')

graph.add_conditional_edges('classification', get_label, {
    'long_answer': 'long_answer',
    'short_answer': 'short_answer'
})

graph.add_edge('long_answer', 'ask_continue')
graph.add_edge('short_answer', 'ask_continue')

graph.add_conditional_edges('ask_continue', should_continue, {
    'replay': 'replay',
    '__end__': END
})

graph.add_edge('replay', 'ask_continue')

app = graph.compile()

print('동물 이름을 입력해 이야기를 시작해보세요!')
user_input = input('동물 이름 입력: ')
final_state = app.invoke({'user_input': user_input})
print('\n대화 종료')