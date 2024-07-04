# 별칭 st 사용
import streamlit as st

# LLM 라이브러리
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


# llm임포트
import llm

# config 임포트
import config

# 환경변수
load_dotenv(dotenv_path=".env")


# =================================== 페이지 url 창 ================================================
# fonfig
# 페이지 url 창에서 제목과 아이콘 설정해줌
st.set_page_config(page_title="소득세 챗봇", page_icon="☆")

# =================================== 페이지 타이틀 ================================================

# 첫번째 열과 두번째 열의 너비를 상대적으로 1:4로 지정
title_col1, title_col2 = st.columns([0.8, 4])

# 첫 번째 열에 이미지 추가
with title_col1:
    st.image("image/robot_icon-icons.com_60269.png", width=100)  # URL 이미지 사용

# 두 번째 열에 제목 추가
with title_col2:
    st.title("소득세 챗봇")

# 캡션 추가
st.caption("소득세에 관련된 모든것을 답해드립니다!")


# =================================== 채팅 input 창 ================================================

# 채팅 입력하고 엔터를 누르면 전체 코드가 다시 돌아감 (리셋됨)
# 따라서 채팅 내용 저장이 필요함.
# 이를 하는 것이 아래
# st.session_state 이 비어있다면, message_list에 빈 리스트 저장
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# message_list 를 돌면서 내용을 출력함

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# if 문 -> 채팅 창 생성
# with 문 -> 글자를 페이지에 띄움
# user 정보와 user_question(질문) 정보를 message_list에 append
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    # 사용자가 입력한 채팅을 페이지에 띄움
    with st.chat_message("user"):
        st.write(user_question) # user_question에 입력한 텍스트 저장
        st.session_state.message_list.append({"role":"user", "content":user_question})

    # 답변 생성 중이라는 ui 출력
    with st.spinner("답변을 생성하는 중입니다."):
        # LLM 모델 답변을 페이지에 띄움
        ai_response=llm.get_ai_response(user_question)  # get_message() 함수를 통해 답변을 얻음
        with st.chat_message("ai"):
            ai_message=st.write_stream(ai_response)
            st.session_state.message_list.append({"role":"ai", "content":ai_message})

# message_list 를 출력
print(st.session_state.message_list)

