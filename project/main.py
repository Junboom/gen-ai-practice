import os
import sys
import io
import torch
import streamlit as st
import fitz  # PyMuPDF

from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph, END

# ----------------------
# 설정
# ----------------------
MODEL_ID = 'torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1'
EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

VECTORSTORE_DIR = 'vectorstore'
IMAGE_DIR = 'images'

CHUNK_SIZE = 1024
TOKEN_SIZE = 512
OVERLAP = 128

PAGE_TITLE = 'Hyundai < i20 > Car Manual QA'

TEMPLATE = '자동차 매뉴얼 문서를 바탕으로 질문에 답해주세요. 반드시 문서 내용만 참고하세요.\n문서 내용:\n{context}\n\n질문:\n{query}\n\n답변:\n'

st.set_page_config(page_title=PAGE_TITLE, layout='wide')

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
# 벡터 스토어 구축
# ----------------------
def build_vectorstore(chunks, page_map):
    docs = [
        Document(page_content=chunk, metadata={"page": page_map[i]})
        for i, chunk in enumerate(chunks)
    ]

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

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
def load_retrieval_qa_chain(retriever):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    text_gen = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map=-1,
        max_new_tokens=TOKEN_SIZE,
        temperature=0.2,
        top_p=0.9
    )

    llm = HuggingFacePipeline(pipeline=text_gen)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return text_gen, qa_chain

def generate_llm_answer(query, qa_chain):
    answers = qa_chain.invoke({'query': query})
    long_answer = answers['result'].split('Helpful Answer:')[-1].strip()
    source_docs = answers.get('source_documents', [])
    unique_docs = {}
    for d in source_docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
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
        key = (d.metadata.get("source"), d.metadata.get("page"))
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
def build_graph(page, vectorstore, text_gen, qa_chain):
    def answer_method(state):
        return {**state, 'page': page}

    def get_page(state):
        return state.get('page', '')

    def llm_answer(state):
        with st.spinner('LLM 답변 생성 중...'):
            query = state.get('query', '')
            answer, source_docs, docs = generate_llm_answer(query, qa_chain)
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
    st.title(PAGE_TITLE)
    page = st.sidebar.radio('답변 방식', ['LLM 답변', 'ChromaDB 검색'])
    st.subheader(f'현재페이지: {page}')

    if len(sys.argv) < 2:
        st.error('PDF 파일 경로를 명령행 인자로 지정해주세요.\n예: streamlit run main.py data/i20.pdf')
        return

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        st.error(f'PDF 파일을 찾을 수 없습니다: {pdf_path}')
        return

    texts, page_images_map = load_pdf(pdf_path)
    chunks, page_map = chunk_text(texts)

    vectorstore, retriever = build_vectorstore(chunks, page_map)
    text_gen, qa_chain = load_retrieval_qa_chain(retriever)
    app = build_graph(page, vectorstore, text_gen, qa_chain)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

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
