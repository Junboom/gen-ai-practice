import os
import sys
import io
import math
import numpy as np
import torch
import pandas as pd
import json
import streamlit as st
import tempfile
import fitz  # PyMuPDF

from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TrainerCallback
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph, END

os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_ENDPOINT']='https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY']='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'
os.environ['LANGCHAIN_PROJECT']='pr-shadowy-scow-41'

# ----------------------
# 설정
# ----------------------
MODEL_ID = 'torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1'
# MODEL_ID = 'tunib/electra-ko-base'
EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

LORA_DIR = 'lora_model'
VECTORSTORE_DIR = 'vectorstore'
IMAGE_DIR = 'images'

DEFAULT_CAR_MENUAL = 'data/i20.pdf'
MY_CAR_STATUS = 'data/used.csv'

CHUNK_SIZE = 1024
TOKEN_SIZE = 512
OVERLAP = 128

PAGE_TITLE = 'Car Manual QA'

TEMPLATE = '자동차 매뉴얼 문서를 바탕으로 질문에 답해주세요.\n문서 내용:\n{context}\n\n질문:\n{query}\n\n답변:\n'

st.set_page_config(page_title=PAGE_TITLE, layout='wide')

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            metrics['eval_perplexity'] = math.exp(metrics['eval_loss'])

# ----------------------
# 텍스트 청크
# ----------------------
def chunk_text(texts):
    chunks = []
    page_map = []

    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        length_function=len
    )

    for page_idx, text in enumerate(texts):
        page_chunks = splitter.split_text(text)
        chunks.extend(page_chunks)
        page_map.extend([page_idx]*len(page_chunks))

    return chunks, page_map

# ----------------------
# CSV 처리
# ----------------------
def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    status_map = {
        -1: '즉시 교체가 필요합니다.',
        0: '곧 교체가 필요할 것 같습니다.',
        1: '교체가 필요하지 않습니다.'
    }

    texts = []
    for _, row in df.iterrows():
        text = (
            f'자동차 소모품: {row["items"]}\n'
            f'현재 주행거리: {row["mileage_km"]} km\n'
            f'현재까지 사용한 개월 수: {row["mileage_month"]}개월\n'
            f'교체 기준 거리: {row["replacement_km"]} km\n'
            f'교체 기준 개월: {row["relacement_month"]}개월\n'
            f'현재 부품의 사용량: {row["used_percent"]}%\n'
            f'교체 시기: {status_map[row["status"]]}'
        )
        texts.append(text)

    return texts

def lora_fine_tuning(base_model, tokenizer, chunks):
    train_texts, eval_texts = train_test_split(chunks, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_dict({'text': train_texts})
    eval_dataset = Dataset.from_dict({'text': eval_texts})

    prepared_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(prepared_model, lora_config)

    def preprocess(data):
        tokenized = tokenizer(
            data['text'],
            truncation=True,
            padding='max_length',
            max_length=TOKEN_SIZE
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized_train = train_dataset.map(preprocess, batched=True)
    tokenized_eval = eval_dataset.map(preprocess, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=LORA_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_perplexity',
        save_total_limit=1,
        greater_is_better=False,
        label_names=['labels'],
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[PerplexityCallback()]
    )

    results = trainer.evaluate()
    eval_loss = results['eval_loss']
    perplexity = math.exp(eval_loss)
    print('Perplexity:', perplexity)

    trainer.train()

    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)

    return model

# ----------------------
# PDF 처리
# ----------------------
def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    page_images_map = {}

    os.makedirs(IMAGE_DIR, exist_ok=True)

    for i, page in enumerate(doc):
        page_text = page.get_text()
        texts.append(page_text)
        images = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image['image']
            img_ext = base_image['ext']
            img_obj = Image.open(io.BytesIO(img_bytes))

            img_path = os.path.join(IMAGE_DIR, f'page{i+1}_{img_index}.{img_ext}')
            img_obj.save(img_path)
            images.append(img_path)

        page_images_map[i] = images

    return texts, page_images_map

# ----------------------
# 벡터 스토어 구축
# ----------------------
def build_vectorstore(chunks, page_map):
    docs = [
        Document(page_content=chunk, metadata={"page": page_map[i]})
        for i, chunk in enumerate(chunks)
    ]

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    if os.path.exists(VECTORSTORE_DIR):
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embedding_model
        )
    else:
        vectorstore = Chroma.from_documents(
            docs,
            embedding_model,
            persist_directory=VECTORSTORE_DIR
        )


    retriever = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding_model
    ).as_retriever(search_kwargs={'k': 3})

    st.success('Chroma Vectorstore 생성 완료')
    return vectorstore, retriever

# ----------------------
# LLM
# ----------------------
def load_pipeline(model, tokenizer):
    return pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map=-1,
        max_new_tokens=TOKEN_SIZE,
        temperature=0.2,
        top_p=0.9
    )

def load_retrieval_qa_chain(text_gen, retriever):
    llm = HuggingFacePipeline(pipeline=text_gen)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def generate_llm_answer(query, text_gen, retriever):
    qa_chain = load_retrieval_qa_chain(text_gen, retriever)
    answers = qa_chain.invoke({'query': query})
    long_answer = answers['result'].split('Helpful Answer:')[-1].strip()
    source_docs = answers.get('source_documents', [])
    unique_docs = {}
    for d in source_docs:
        key = (d.metadata.get('source'), d.metadata.get('page'))
        if key not in unique_docs:
            unique_docs[key] = d
    source_docs = list(unique_docs.values())
    docs = [doc.page_content for doc in source_docs]
    return long_answer, source_docs, docs

def generate_pdf_answer(query, vectorstore, text_gen):
    prompt = PromptTemplate.from_template(TEMPLATE)
    source_docs = vectorstore.similarity_search(query, k=3)
    unique_docs = {}
    for d in source_docs:
        key = (d.metadata.get('source'), d.metadata.get('page'))
        if key not in unique_docs:
            unique_docs[key] = d
    source_docs = list(unique_docs.values())
    docs = [doc.page_content for doc in source_docs]
    context = '\n'.join(docs)
    formatted_prompt = prompt.format(context=context, query=query)
    output = text_gen(formatted_prompt, max_new_tokens=TOKEN_SIZE)[0]['generated_text']
    short_answer = output.split('답변:')[-1].strip()
    return short_answer, source_docs, docs

def generate_summary(docs, text_gen):
    prompt = PromptTemplate.from_template('다음 문서 내용을 바탕으로 요약해주세요.\n문서 내용:\n{context}\n\n요약:\n')
    formatted_prompt = prompt.format(context=docs)
    output = text_gen(formatted_prompt, max_new_tokens=TOKEN_SIZE)[0]['generated_text']
    summary = output.split('요약:')[-1].strip()
    return summary

# ----------------------
# UI
# ----------------------
def display_answer(answer, summary, docs, images):
    with st.chat_message('assistant'):
        st.markdown('### 답변')
        st.markdown(answer)

        if summary:
            st.markdown('### 요약')
            st.markdown(summary)

        st.markdown('### 관련 문서')
        for i, context in enumerate(docs, 1):
            st.markdown(f'{i}. {context}')

        if images:
            st.markdown('### 관련 이미지')
            cols = st.columns(3)
            for i, img_path in enumerate(images):
                with cols[i%3]:
                    st.image(img_path, use_container_width=True)

# ----------------------
# langgraph
# ----------------------
def build_graph(page, vectorstore, text_gen, retriever):
    def answer_method(state):
        return {**state, 'page': page}

    def get_page(state):
        return state.get('page', '')

    def llm_answer(state):
        with st.spinner('LLM 답변 생성 중...'):
            query = state.get('query', '')
            answer, source_docs, docs = generate_llm_answer(query, text_gen, retriever)
        return {**state, 'answer': answer, 'source_docs': source_docs, 'docs': docs}

    def pdf_answer(state):
        with st.spinner('ChromaDB 검색으로 답변 생성 중...'):
            query = state.get('query', '')
            answer, source_docs, docs = generate_pdf_answer(query, vectorstore, text_gen)
        return {**state, 'answer': answer, 'source_docs': source_docs, 'docs': docs}

    def next_step(state):
        return {**state}

    def need_summary(state):
        answer = state.get('answer', '')
        answer_len = len(answer)
        st.markdown(f'답변 길이: {answer_len}')
        return 'get_summary' if answer_len > TOKEN_SIZE else END

    def get_summary(state):
        docs = state.get('docs', '')
        summary = generate_summary(docs, text_gen)
        return {**state, 'summary': summary}

    graph = StateGraph(dict)

    graph.add_node('answer_method', answer_method)
    graph.add_node('llm_answer', llm_answer)
    graph.add_node('pdf_answer', pdf_answer)
    graph.add_node('next_step', next_step)
    graph.add_node('get_summary', get_summary)

    graph.set_entry_point('answer_method')
    graph.add_conditional_edges('answer_method', get_page, {
        'LLM 답변': 'llm_answer',
        'ChromaDB 검색': 'pdf_answer'
    })

    graph.add_edge('llm_answer', 'next_step')
    graph.add_edge('pdf_answer', 'next_step')

    graph.add_conditional_edges('next_step', need_summary, {
        'get_summary': 'get_summary',
        END: END
    })

    graph.add_edge('get_summary', END)

    return graph.compile()

# ----------------------
# main
# ----------------------
def main():
    st.sidebar.title('Options')

    pdf_path = DEFAULT_CAR_MENUAL
    uploaded_file = st.sidebar.file_uploader('차종 변경', type='pdf')
    if uploaded_file is not None:
        pdf_path = f'./data/{uploaded_file.name}'

    page = st.sidebar.radio('답변 방식', ['LLM 답변', 'ChromaDB 검색'])

    file_name = os.path.basename(pdf_path)
    car_name = os.path.splitext(file_name)[0]
    automobile = 'Hyundai' if car_name == 'i20' else 'Mercedes Benz'
    prefix = f'{automobile} < {car_name} >'

    st.title(f'{prefix} {PAGE_TITLE}')
    st.markdown(f'현재 차종: **{car_name}** / 답변 방식: **{page}**')

    texts, page_images_map = load_pdf(pdf_path)
    chunks, page_map = chunk_text(texts)

    vectorstore, retriever = build_vectorstore(chunks, page_map)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if os.path.exists(LORA_DIR):
        model = PeftModel.from_pretrained(model, LORA_DIR)

    if st.sidebar.button('차량 점검하기'):
        with st.spinner('LoRA fine-tuning 기반 차량 점검 중...'):
            texts = load_csv(MY_CAR_STATUS)
            chunks, _ = chunk_text(texts)
            model = lora_fine_tuning(model, tokenizer, chunks)
            st.write('차량 점검 완료')

    text_gen = load_pipeline(model, tokenizer)

    app = build_graph(page, vectorstore, text_gen, retriever)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        if message['role'] == 'user':
            with st.chat_message('user'):
                st.markdown(message['content'])
        else:
            content = message['content']
            display_answer(
                content['answer'],
                content.get('summary', ''),
                content.get('docs', []),
                content.get('images', [])
            )

    query = st.chat_input('질문 입력')
    if query:
        st.session_state.messages.append({'role': 'user', 'content': query})
        with st.chat_message('user'):
            st.markdown(query)

        with st.spinner('답변 생성 중...'):
            results = app.invoke({'query': query})
            answer = results['answer']
            summary = results.get('summary', '')
            source_docs = results.get('source_docs', [])
            docs = results['docs']
            images = []
            for doc in source_docs:
                page_idx = doc.metadata.get('page')
                if page_idx is not None:
                    images.extend(page_images_map.get(page_idx, []))

        st.session_state.messages.append({
            'role': 'assistant',
            'content': {
                'answer': answer,
                'summary': summary,
                'docs': docs,
                'images': images
            }
        })
        display_answer(answer, summary, docs, images)

if __name__ == "__main__":
    main()
