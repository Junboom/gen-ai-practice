import streamlit as st

st.title('제목을 사용하는 함수')
st.header('1. 헤더 입력')
st.subheader('2. 서브 헤더')

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('챗봇 데모')
st.write('대화형 챗봇 입니다.')

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

user_input = st.chat_input('메세지를 입력하세요')

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    bot_response = f'저도 따라할래요 "{user_input}"'

    st.session_state.messages.append({'role': 'assistant', 'content': bot_response})
    with st.chat_message('assistant'):
        st.markdown(bot_response)

st.header('6. 파일 업로드 및 다운로드')

uploaded_file = st.file_uploader('파일을 업로드하세요.', type=['csv', 'txt', 'png', 'jpg'])
if uploaded_file is not None:
    st.write('업로드한 파일 이름: ', uploaded_file)

st.download_button(label='샘플 다운로드', data='hello, streamlit!', file_name='sample.txt')

st.sidebar.header('사이드바 메뉴')
page = st.sidebar.radio('이동할 페이지 선택', ['홈', '데이터', '설정'])
st.write(f'현재페이지: {page}')

st.header('3. 버튼 및 이벤트')

if st.button('클릭하세요'):
    st.write('버튼이 클릭되었습니다!')

toggle_state = st.toggle('토글 스위치')
st.write(f'토글 상태: {toggle_state}')

st.write('마지막까지 파이팅!')

st.header('2. 사용자 입력 받기')

name = st.text_input('이름을 입력하세요: ')
st.write(f'입력한 이름: {name}')

name = st.text_area('자기 소개: ')
st.write(f'자기 소개: {name}')

name = st.number_input('나이를 입력하세요: ', min_value=0, max_value=100, step=1)
st.write(f'입력한 나이: {name}')

name = st.date_input('생년월일을 입력하세요: ')
st.write(f'입력한 날짜: {name}')

name = st.selectbox('좋아하는 색상을 선택하세요: ', ['빨강', '파랑', '초록', '노랑'])
st.write(f'선택한 색상: {name}')

name = st.multiselect('취미를 선택하세요: ', ['독서', '운동', '게임', '요리'])
st.write(f'선택한 취미: {name}')

name = st.radio('성별을 선택하세요: ', ['남성', '여성', '비공개'])
st.write(f'입력한 이름: {name}')

name = st.checkbox('이용 약관에 동의합니다.')
if name:
    st.write(f'약관에 동의하셨습니다.')

score = st.slider('점수를 선택하세요: ', min_value=0, max_value=100)
st.write(f'선택한 점수: {score}')