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

# ----------------------
# 설정
# ----------------------
MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

VECTORSTORE_DIR = 'vectorstore'
IMAGE_DIR = 'images'
THUMB_DIR = 'thumbnails'

CHUNK_SIZE = 512
OVERLAP = 64

PAGE_TITLE = 'Hyundai < i20 > Car Manual QA'

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
def build_vectorstore(page_images_map, chunks, page_map):
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(page_content=chunk, metadata={'page': page_map[i]}))

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        docs,
        embedding_model,
        persist_directory=VECTORSTORE_DIR
    )

    st.success('Chroma Vectorstore 생성 완료')
    return vectorstore, page_images_map

# ----------------------
# 검색
# ----------------------
def search_vectorstore(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    return results  # list of Documents

# ----------------------
# LLM
# ----------------------
def load_retrieval_qa_chain(vectorstore):
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

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    prompt = PromptTemplate.from_template(
        '당신은 PDF를 기반으로 답변을 제공하는 AI입니다.\n\n본문:\n{context}\n\n질문:\n{question}\n\n답변:\n'
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={'prompt': prompt},
        return_source_documents=True
    )

    return qa_chain

def generate_answer(qa_chain, query, page_images_map):
    result = qa_chain.invoke(query)

    short_answer = result['result'].strip()
    source_docs = result.get('source_documents', [])

    context_texts = [doc.page_content for doc in source_docs]

    # 관련 이미지
    related_images = []
    for doc in source_docs:
        page_idx = doc.metadata.get('page')
        if page_idx is not None:
            related_images.extend(page_images_map.get(page_idx, []))

    return short_answer, context_texts, related_images

# ----------------------
# UI
# ----------------------
def display_answer(ans, docs, images):
    st.markdown('### 답변')
    st.markdown(ans)

    st.markdown('### 관련 문서 조각')
    for i, doc in enumerate(docs,1):
        st.markdown(f"{i}. {doc}")

    if images:
        st.markdown('### 관련 이미지')
        cols = st.columns(3)
        for i, img_path in enumerate(images):
            with cols[i%3]:
                st.image(img_path, use_container_width=True)

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

    if os.path.exists(VECTORSTORE_DIR):
        docs = [
            Document(page_content=chunk, metadata={"page": page_map[i]})
            for i, chunk in enumerate(chunks)
        ]
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=VECTORSTORE_DIR
        )
    else:
        vectorstore, page_images_map = build_vectorstore(page_images_map, chunks, page_map)

    qa_chain = load_retrieval_qa_chain(vectorstore)

    query = st.text_input('질문 입력')
    if st.button('질문하기') and query:
        with st.spinner('답변 생성 중...'):
            ans, docs, images = generate_answer(qa_chain, query, page_images_map)
        display_answer(ans, docs, images)

if __name__ == "__main__":
    main()
