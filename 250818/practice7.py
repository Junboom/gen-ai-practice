import os

os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_ENDPOINT']='https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY']='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'
os.environ['LANGCHAIN_PROJECT']='pr-shadowy-scow-41'

from langgraph.graph import StateGraph, END

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

def get_label(state):
    return state.get('label', '')

graph = StateGraph(dict)

graph.add_node('classification', classification)
graph.add_node('positive', positive_answer)
graph.add_node('negative', negative_answer)
graph.add_node('neutral', neutral_answer)

graph.set_entry_point('classification')
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