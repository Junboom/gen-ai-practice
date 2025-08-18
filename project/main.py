import os
import sys
import io
import torch
import streamlit as st
import re
import fitz  # PyMuPDF

from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# LangChain
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# ----------------------
# 설정
# ----------------------
MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

CHUNK_SIZE = 512
OVERLAP = 64

VECTORSTORE_DIR = 'vectorstore'
IMAGE_DIR = 'images'
THUMB_DIR = 'thumbnails'

PAGE_TITLE = 'Hyundai i20 Car Manual QA'

st.set_page_config(page_title=PAGE_TITLE, layout='wide')

# ----------------------
# PDF 처리
# ----------------------
def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    page_images_map = {}  # 페이지 번호 -> 이미지 리스트
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)

    for i, page in enumerate(doc):
        page_text = page.get_text()
        texts.append(page_text)
        images = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_obj = Image.open(io.BytesIO(img_bytes))

            img_path = os.path.join(IMAGE_DIR, f"page{i+1}_{img_index}.{img_ext}")
            img_obj.save(img_path)
            images.append(img_path)

            thumb_path = os.path.join(THUMB_DIR, f"thumb_{i+1}_{img_index}.{img_ext}")
            img_obj.thumbnail((200,200))
            img_obj.save(thumb_path)

        page_images_map[i] = images

    return texts, page_images_map

# ----------------------
# 텍스트 청크
# ----------------------
def chunk_text(texts, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    page_map = []
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
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
def build_vectorstore(pdf_path):
    texts, page_images_map = load_pdf(pdf_path)
    chunks, page_map = chunk_text(texts)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(page_content=chunk, metadata={"page": page_map[i]}))

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory=VECTORSTORE_DIR)
    vectorstore.persist()

    st.success("✅ Chroma Vectorstore 생성 완료")
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
@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    device_map="auto", temperature=0.7, top_p=0.9)
    return pipe

def generate_answer(pipe, query, vectorstore, page_images_map):
    search_results = search_vectorstore(vectorstore, query)
    context_texts = [doc.page_content for doc in search_results]
    context = "\n".join(context_texts)

    prompt = f"""
자동차 매뉴얼 문서를 바탕으로 질문에 답해주세요. 반드시 문서 내용만 참고하세요.
문서 내용:
{context}

질문: {query}
답변:
"""
    output = pipe(prompt, max_new_tokens=600)[0]["generated_text"]
    answer_text = output.split(prompt)[-1].strip()

    # 관련 이미지
    related_images = []
    for doc in search_results:
        page_idx = doc.metadata.get("page")
        if page_idx is not None:
            related_images.extend(page_images_map.get(page_idx, []))

    return answer_text, context_texts, related_images

# ----------------------
# UI
# ----------------------
def display_answer(ans, docs, images):
    st.markdown("### 💬 답변")
    st.markdown(ans)
    st.markdown("### 📄 관련 문서 조각")
    for i, doc in enumerate(docs,1):
        st.markdown(f"{i}. {doc}")

    if images:
        st.markdown("### 🖼️ 관련 이미지")
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
        st.error("❌ PDF 파일 경로를 명령행 인자로 지정해주세요.\n예: streamlit run main.py data/i20.pdf")
        return
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        st.error(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return

    # Vectorstore 로드 또는 생성
    if os.path.exists(VECTORSTORE_DIR):
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding_model)
        _, page_images_map = load_pdf(pdf_path)
    else:
        vectorstore, page_images_map = build_vectorstore(pdf_path)

    llm_pipe = load_llm_model()

    query = st.text_input("질문 입력")
    if st.button("질문하기") and query:
        with st.spinner("답변 생성 중..."):
            ans, docs, images = generate_answer(llm_pipe, query, vectorstore, page_images_map)
        display_answer(ans, docs, images)

if __name__ == "__main__":
    main()
