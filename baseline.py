import os
import unicodedata

import torch
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from accelerate import Accelerator

# Langchain 관련
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

device = torch.device('mps')

def process_pdf(file_path, chunk_size=800, chunk_overlap=50):
    """PDF 텍스트 추출 후 chunk 단위로 나누기"""
    # PDF 파일 열기
    doc = fitz.open(file_path)
    text = ''
    # 모든 페이지의 텍스트 추출
    for page in doc:
        text += page.get_text()
    # 텍스트를 chunk로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunk_temp = splitter.split_text(text)
    # Document 객체 리스트 생성
    chunks = [Document(page_content=t) for t in chunk_temp]
    return chunks


def create_vector_db(chunks, model_path="intfloat/multilingual-e5-small"):
    """FAISS DB 생성"""
    # 임베딩 모델 설정
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # FAISS DB 생성 및 반환
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db

def normalize_path(path):
    """경로 유니코드 정규화"""
    return unicodedata.normalize('NFC', path)


def process_pdfs_from_dataframe(df, base_directory):
    """딕셔너리에 pdf명을 키로해서 DB, retriever 저장"""
    pdf_databases = {}
    unique_paths = df['Source_path'].unique()
    
    for path in tqdm(unique_paths, desc="Processing PDFs"):
        # 경로 정규화 및 절대 경로 생성
        normalized_path = normalize_path(path)
        full_path = os.path.normpath(os.path.join(base_directory, normalized_path.lstrip('./'))) if not os.path.isabs(normalized_path) else normalized_path
        
        pdf_title = os.path.splitext(os.path.basename(full_path))[0]
        print(f"Processing {pdf_title}...")
        
        # PDF 처리 및 벡터 DB 생성
        chunks = process_pdf(full_path)
        db = create_vector_db(chunks)
        
        # Retriever 생성
        retriever = db.as_retriever(search_type="mmr", 
                                    search_kwargs={'k': 3, 'fetch_k': 8})
        
        # 결과 저장
        pdf_databases[pdf_title] = {
                'db': db,
                'retriever': retriever
        }
    return pdf_databases

base_directory = './open' # Your Base Directory
df = pd.read_csv('./open/test.csv')
pdf_databases = process_pdfs_from_dataframe(df, base_directory)


def setup_llm_pipeline():
    # 4비트 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 모델 ID 
    model_id = "beomi/llama-2-ko-7b"

    # 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False

    # 모델 로드 및 양자화 설정 적용
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True )

    # HuggingFacePipeline 객체 생성
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        return_full_text=False,
        max_new_tokens=128,
    )

    hf = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return hf

# LLM 파이프라인
llm = setup_llm_pipeline()

def normalize_string(s):
    """유니코드 정규화"""
    return unicodedata.normalize('NFC', s)

def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    context = ""
    for doc in docs:
        context += doc.page_content
        context += '\n'
    return context

# 결과를 저장할 리스트 초기화
results = []

# DataFrame의 각 행에 대해 처리
for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering Questions"):
    # 소스 문자열 정규화
    source = normalize_string(row['Source'])
    question = row['Question']

    # 정규화된 키로 데이터베이스 검색
    normalized_keys = {normalize_string(k): v for k, v in pdf_databases.items()}
    retriever = normalized_keys[source]['retriever']

    # RAG 체인 구성
    template = """
    다음 정보를 바탕으로 질문에 답하세요:
    {context}

    질문: {question}

    답변:
    """
    prompt = PromptTemplate.from_template(template)

    # RAG 체인 정의
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 답변 추론
    print(f"Question: {question}")
    full_response = rag_chain.invoke(question)

    print(f"Answer: {full_response}\n")

    # 결과 저장
    results.append({
        "Source": row['Source'],
        "Source_path": row['Source_path'],
        "Question": question,
        "Answer": full_response
    })

# 제출용 샘플 파일 로드
submit_df = pd.read_csv("./sample_submission.csv")

# 생성된 답변을 제출 DataFrame에 추가
submit_df['Answer'] = [item['Answer'] for item in results]
submit_df['Answer'] = submit_df['Answer'].fillna("데이콘")     # 모델에서 빈 값 (NaN) 생성 시 채점에 오류가 날 수 있음 [ 주의 ]

# 결과를 CSV 파일로 저장
submit_df.to_csv("./baseline_submission.csv", encoding='UTF-8-sig', index=False)