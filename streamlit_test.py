# 별칭 st 사용
import streamlit as st

# 페이지 url 창 구성
st.set_page_config(
    page_title="My Page", 
    page_icon="image/cat.png"
)


# 페이지 제목 구성
# 첫번째 열과 두번째 열의 너비를 상대적으로 1:4로 지정
title_col1, title_col2 = st.columns([1, 4])

# 첫 번째 열에 이미지 추가
with title_col1:
    st.image("image/cat.png", width=100)  # URL 이미지 사용

# 두 번째 열에 제목 추가
with title_col2:
    st.title("고양이 페이지")

# 글 추가
#  HTML과 CSS로 글씨 크기를 조정한 캡션 추가
st.markdown("<p style='font-size:25px;'>고양이 페이지입니다</p>", unsafe_allow_html=True)


# st.session_state 이 비어있다면, message_list에 빈 리스트 저장
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# message_list 를 돌면서 내용을 출력함
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 사용자 입력창 추가
if user_question := st.chat_input(placeholder="아무말이나 입력하세요"):
    with st.chat_message("user"):
        st.write(user_question)
        st.session_state.message_list.append({"role":"user", "content":user_question}) # list에 추가함

    # 답변 생성 중이라는 ui 출력
    with st.spinner("답변을 생성하는 중입니다."):
        # LLM 모델 답변을 페이지에 띄움
        with st.chat_message("ai"):
            return_message="안녕하세요 챗봇입니다."
            st.write(return_message)
            st.session_state.message_list.append({"role":"ai", "content":return_message})
