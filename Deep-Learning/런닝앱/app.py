import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import numpy as np
import joblib
# import xgboost as xgb_lib # XGBoost crashes on Streamlit/Mac

# --- 1. 모델 아키텍처 정의 (PyTorch) ---
class ChurnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 64) 
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# --- 2. 모델 및 스케일러 로드 ---
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    rf_model = joblib.load('rf_model.pkl')
    
    # [Fix] XGBoost removed due to stability issues
    # xgb_model = xgb_lib.Booster()
    # xgb_model.load_model('xgb_model.json')
    xgb_model = None
    
    dl_model = ChurnModel()
    dl_model.load_state_dict(torch.load('dl_model.pth'))
    dl_model.eval()
    
    return scaler, rf_model, xgb_model, dl_model

try:
    scaler, rf, xgb, dl = load_assets()
    models_loaded = True
except:
    models_loaded = False
    st.error("모델 파일을 찾을 수 없습니다. 먼저 running.ipynb를 실행하여 모델을 저장해주세요.")

# --- 3. UI 디자인 ---
st.set_page_config(page_title="러닝 앱 이탈 방지 코치", page_icon="🏃‍♂️")

st.title("🏃‍♂️ Running Churn Defender")
st.markdown("""
**"회원님이 런닝을 그만둘까? AI 코치가 진단해드립니다!"**  
좌측 사이드바에서 회원님의 현재 상태를 입력해보세요.
""")

if models_loaded:
    with st.sidebar:
        st.header("📋 회원 정보 입력")
        
        st.subheader("1. 활동량")
        run_count = st.slider("총 런닝 횟수 (Run Count)", 5, 500, 50)
        
        st.subheader("2. 오버페이스 (무리함)")
        overpace_ratio = st.slider("오버페이스 비율 (0.0~1.0)", 0.0, 1.0, 0.1)
        recent_overpace = st.slider("최근 5회 중 오버페이스 횟수", 0, 5, 0)
        
        st.subheader("3. 회복과 패턴")
        avg_recovery = st.slider("평균 휴식 간격 (일)", 1.0, 30.0, 3.0)
        consistency = st.slider("휴식 간격의 불규칙성 (낮을수록 좋음)", 0.0, 50.0, 5.0)
        
        st.subheader("4. 지루함과 흥미")
        routine_monotony = st.slider("루틴 단조로움 (시간 편차, 낮을수록 지루함)", 0.0, 30.0, 10.0)
        interest_decay = st.slider("흥미 감소도 (최근 간격 급증, 1.0 이상 위험)", 0.5, 5.0, 1.0)

        st.markdown("---")
        model_choice = st.selectbox("사용할 AI 모델 선택", ["Deep Learning (PyTorch)", "Random Forest"])

    # --- 4. 예측 로직 ---
    # 입력 데이터 배열
    input_data = np.array([[run_count, overpace_ratio, recent_overpace, avg_recovery, consistency, routine_monotony, interest_decay]])
    input_scaled = scaler.transform(input_data)

    churn_prob = 0.0
    
    if model_choice == "Deep Learning (PyTorch)":
        with torch.no_grad():
            tensor_in = torch.FloatTensor(input_scaled)
            churn_prob = dl(tensor_in).item()
    else: # Random Forest
        churn_prob = rf.predict_proba(input_scaled)[0][1]

    # --- 5. 결과 대시보드 ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(label="이탈 위험도", value=f"{churn_prob*100:.1f}%")
        
        if churn_prob > 0.7:
            st.error("위험 (Danger)")
        elif churn_prob > 0.4:
            st.warning("주의 (Warning)")
        else:
            st.success("안전 (Safe)")

    with col2:
        st.subheader("🤖 AI 코치의 처방전")
        
        # 시나리오별 조언 (우선순위 로직)
        advice_given = False
        
        if interest_decay >= 2.0:
            st.warning("🥀 **[권태기 감지]** 예전보다 운동 간격이 2배 이상 길어졌어요! '가볍게 10분만 뛰기'로 다시 습관을 잡아볼까요?")
            advice_given = True
            
        if routine_monotony < 5.0:
            st.info("🥱 **[지루함 주의]** 매번 똑같은 패턴으로 운동하시네요. 새로운 러닝 코스나 플레이리스트가 필요합니다!")
            advice_given = True
            
        if recent_overpace >= 3:
            st.error("🔥 **[부상 경고]** 최근 5번 중 3번이나 무리하셨어요. 당장 속도를 줄이지 않으면 100% 다칩니다.")
            advice_given = True
            
        if not advice_given:
            if churn_prob > 0.5:
                st.write("💪 조금만 더 힘내세요! 꾸준함이 정답입니다.")
            else:
                st.balloons()
                st.write("🌟 **완벽합니다!** 페이스 조절도 훌륭하고 꾸준하시네요. 슈퍼 러너가 될 자질이 보입니다.")

    # --- 6. 상세 분석 (Expander) ---
    with st.expander("📊 상세 분석 데이터 보기"):
        st.write("입력된 데이터 (Scaled):", input_scaled)
        st.info(f"사용된 모델: {model_choice}")
