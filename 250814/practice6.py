from langchain.memory import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from uuid import uuid4

model_id = 'skt/kogpt2-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

text_gen = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    truncation=False,
    max_new_tokens=64,
    do_sample=True,
    return_full_text=False,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=text_gen)

prompt = PromptTemplate.from_template(
    # '지금까지의 이야기:\n{history}\n\n이제 "{animal}"에 대한 다음 이야기를 써주세요:\n'
    '{history}\n{animal}\n'
)

chat_histories = {}

def get_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def run():
    session_id = str(uuid4())
    print('동물 이름을 입력해 이야기를 시작하세요.')

    while True:
        animal = input('\n동물 이름 또는 질문 입력 (종료하려면 "종료"): ').strip()
        if animal in ['종료', 'exit', 'quit']:
            print('대화를 종료합니다.')
            break

        history = get_history(session_id)

        if history.messages:
            history_text = '\n'.join(
                f'{msg.type.upper()}: {msg.content}' for msg in history.messages
            )
            prompt_input = prompt.format(history=history_text, animal=animal)
        else:
            prompt_input = f'"아주 먼 옛날 {animal} 한 마리가 살고 있었습니다. 그러던 어느날 "'
        
        response = llm.invoke(prompt_input)

        print('\n생성된 이야기:')
        print(response)

        history.add_user_message(animal)
        history.add_ai_message(response)

run()