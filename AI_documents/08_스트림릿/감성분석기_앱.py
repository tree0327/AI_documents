import streamlit as st
import random

# 페이지 설정
st.set_page_config(page_title="AI 감성 분석기", page_icon="💖")

# 제목 및 설명
st.title("🤖 내 마음을 읽는 AI")
st.write("오늘 있었던 일을 적어주세요. AI가 당신의 기분을 분석해드립니다! (비유: 마음의 날씨 예보 ☀️☔️)")

# 사용자 입력 받기
user_input = st.text_area("여기에 일기를 써보세요:", height=150)

# 간단한 감성 분석 함수 (규칙 기반)
def analyze_sentiment(text):
    positive_words = ["행복", "좋아", "기뻐", "즐거", "사랑", "감사", "최고", "성공"]
    negative_words = ["슬퍼", "화나", "짜증", "우울", "힘들", "아파", "망했", "실패"]
    
    score = 0
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1
            
    return score

# 버튼 클릭 시 실행
if st.button("분석 시작! 🚀"):
    if user_input:
        with st.spinner("AI가 고민 중입니다..."):
            score = analyze_sentiment(user_input)
            
            st.divider()
            
            if score > 0:
                st.success("🌈 오늘은 '맑음'이네요! 긍정적인 에너지가 느껴져요.")
                st.balloons()
            elif score < 0:
                st.error("🌧️ 오늘은 '비'가 내리네요. 따뜻한 코코아 한 잔 어때요?")
            else:
                st.info("☁️ 오늘은 '흐림'이군요. 차분하게 하루를 정리해보세요.")
                
            st.write(f"**감성 점수**: {score}점")
    else:
        st.warning("내용을 입력해주세요!")

# 사이드바 (추가 정보)
with st.sidebar:
    st.header("사용법 💡")
    st.write("1. 텍스트 창에 글을 씁니다.")
    st.write("2. '분석 시작' 버튼을 누릅니다.")
    st.write("3. AI의 피드백을 확인합니다.")
    st.divider()
    st.write("Made with Streamlit")
