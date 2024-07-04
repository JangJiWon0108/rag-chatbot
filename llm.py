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

# config 임포트
import config


# 환경변수
load_dotenv(dotenv_path=".env")

# =================================== LLM 모델 로딩 함수 정의 ================================================
def get_llm():
    # LLM 모델 가져오기 
    llm=ChatUpstage(api_key="up_D8sLGDa0x7FknzDfrMuiVMxEObmEP")

    return llm

# =================================== 임베딩 모델, DB 가져오고, Retrieval 정의하는 함수 정의 ================================================
def get_retrieval():
    # 임베딩 모델 가져오기
    embedding=UpstageEmbeddings(model="solar-embedding-1-large")
    # Pinecone DB 가져오기
    index_name="tax-index"
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name, 
        embedding=embedding
    )
    # retrieval 정의
    retrieval=database.as_retriever(search_kwargs={"k":2})

    return retrieval
# =================================== 임베딩 모델, DB 가져오고, Retrieval 정의하는 함수 정의 ================================================

def get_history_retrieval():
    # retrieval 가져옴
    retrieval=get_retrieval()

    # llm 가져옴
    llm=get_llm()

    # 프롬프트 가져오기
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # history 리트리버 정의
    history_aware_retriever = create_history_aware_retriever(
        llm, retrieval, contextualize_q_prompt
    )

    # 리턴
    return history_aware_retriever

# =================================== 딕셔너리 chain 함수 정의 ================================================
def get_dictionary_chain():
    # 키워드사전 정의
    dictionary=["사람을 나타내는 모든 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
                                            
        사전 : {dictionary}
        
        질문 : {{question}}
                                            
    """)
    # llm 불러옴
    llm=get_llm()

    # 딕셔너리 체인
    dictionary_chain = prompt | llm | StrOutputParser()

    # 딕셔너리 체인 리턴
    return dictionary_chain


# =================================== RAG Chain 함수 정의 ================================================
def get_rag_chain():
    # llm, retrieval 가져옴
    llm=get_llm()

    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    # few shot 프롬프트 정의
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=config.answer_examples,  # config.py 에서 정의한 예시 지정
    )

    print(few_shot_prompt.invoke({}).to_messages())


    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요."
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 답변을 원합니다"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt, # few_shot 프롬프트 지정
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # history_aware_retriever 가져옴
    history_aware_retriever=get_history_retrieval()

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # store 변수에 history 저장
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain


# =================================== LLM 답을 받는 함수 정의 ================================================
def get_ai_response(user_message):

    # 딕셔너리 체인
    dictionary_chain=get_dictionary_chain()

    # qa 체인
    qa_chain=get_rag_chain()

    # 두 체인 결합
    tax_chain={"input":dictionary_chain} | qa_chain

    # stream 로 질문 받기
    # invoke -> stream 으로 바꾸면 iterator 로 바뀜
    ai_message=tax_chain.stream(
        {
            "question":user_message
        },
        config={
        "configurable": {"session_id": "abc123"}
        },
    )
    
    # 리턴
    return ai_message
