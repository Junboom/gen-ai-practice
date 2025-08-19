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
THUMB_DIR = 'thumbnails'

CHUNK_SIZE = 512
OVERLAP = 64

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
    os.makedirs(THUMB_DIR, exist_ok=True)

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

            thumb_path = os.path.join(THUMB_DIR, f'thumb_{i+1}_{img_index}.{img_ext}')
            img_obj.thumbnail((200,200))
            img_obj.save(thumb_path)

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
    ).as_retriever(search_kargs={'k': 3})

    st.success('Chroma Vectorstore 생성 완료')
    return vectorstore, retriever

# ----------------------
# 검색
# ----------------------
def search_vectorstore(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    return results  # list of Documents

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
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9
    )

    llm = HuggingFacePipeline(pipeline=text_gen)

    prompt = PromptTemplate.from_template(TEMPLATE)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return text_gen, qa_chain

def generate_long_answer(query):
    answers = qa_chain.invoke({'query': query})
    long_answer = answers['result'].split('답변:')[-1].strip()

    source_docs = answers.get('source_documents', [])
    docs = [doc.page_content for doc in source_docs]

    return long_answer, source_docs, docs

def generate_short_answer(text_gen, vectorstore, query):
    output = text_gen(TEMPLATE, max_new_tokens=256)[0]['generated_text']
    short_answer = output.split(TEMPLATE)[-1].strip()

    source_docs = search_vectorstore(vectorstore, query)
    docs = [doc.page_content for doc in source_docs]

    return short_answer, source_docs, docs

# ----------------------
# UI
# ----------------------
def display_answer(answer, docs, images):
    with st.chat_message('assistant'):
        st.markdown('### 답변')
        st.markdown(answer)

        st.markdown('### 관련 문서 조각')
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
def build_graph(vectorstore, retriever, text_gen, qa_chain):
    def classification(state):
        query = state.get('query', '')
        output = text_gen(TEMPLATE, max_new_tokens=256)[0]['generated_text']
        output_len = len(output)
        st.markdown(f'답변 길이: {output_len}')
        label = 'long_answer' if output_len > 256 else 'short_answer'
        return {**state, 'label': label}

    def long_answer(state):
        with st.spinner('긴 답변 생성 중...'):
            query = state.get('query', '')
            answer, source_docs, docs = generate_long_answer(query)
        return {**state, 'answer': answer, 'source_docs': source_docs, 'docs': docs}

    def short_answer(state):
        with st.spinner('짧은 답변 생성 중...'):
            query = state.get('query', '')
            answer, source_docs, docs = generate_short_answer(text_gen, vectorstore, query)
        return {**state, 'answer': answer, 'source_docs': source_docs, 'docs': docs}

    def ask_continue(state):
        return {**state, 'continue': False}

    def should_continue(state):
        return 'ask_continue' if state.get('continue') else END

    def get_label(state):
        return state.get('label', '')

    graph = StateGraph(dict)

    graph.add_node('classification', classification)
    graph.add_node('long_answer', long_answer)
    graph.add_node('short_answer', short_answer)
    graph.add_node('ask_continue', ask_continue)

    graph.set_entry_point('classification')

    graph.add_conditional_edges('classification', get_label, {
        'long_answer': 'long_answer',
        'short_answer': 'short_answer'
    })

    graph.add_edge('long_answer', 'ask_continue')
    graph.add_edge('short_answer', 'ask_continue')

    graph.add_conditional_edges('ask_continue', should_continue, {
        'ask_continue': 'ask_continue',
        END: END
    })

    return graph.compile()

# ----------------------
# main
# ----------------------
def main():
    st.title(PAGE_TITLE)

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
    app = build_graph(vectorstore, retriever, text_gen, qa_chain)

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
            source_docs = results.get('source_docs', [])
            docs = results['docs']
            images = []
            for doc in source_docs:
                page_idx = doc.metadata.get('page')
                if page_idx is not None:
                    images.extend(page_images_map.get(page_idx, []))

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        display_answer(answer, docs, images)

if __name__ == "__main__":
    main()
