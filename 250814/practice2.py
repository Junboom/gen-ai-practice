from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=text_gen)

prompt = '산 속에 토끼 한 마리가 살고 있었습니다. 그러던 어느 날 '
response = llm.invoke(prompt)

print('생성된 문장: ', response)