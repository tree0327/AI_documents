import streamlit as st
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# 1. 페이지 설정
st.set_page_config(page_title="💰 AI 주식 투자 비서", page_icon="📈")
st.title("🤖 내일의 주가를 예측해드립니다!")
st.write("딥러닝(LSTM) 모델이 지난 차트를 분석해서 미래를 예측합니다.")

# 2. 사용자 입력 (종목 코드)
stock_code = st.text_input("종목 코드를 입력하세요 (예: 005930 삼성전자)", "005930")

if st.button("예측 시작! 🚀"):
    with st.spinner("최신 데이터를 분석 중입니다..."):
        # 3. 데이터 가져오기 (최근 100일)
        df = fdr.DataReader(stock_code, "2023-01-01")

        if len(df) < 60:
            st.error("데이터가 너무 적습니다. 다른 종목을 선택해주세요.")
        else:
            # 4. 데이터 전처리
            data = df[["Close"]].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # 최근 10일치 데이터로 다음날 예측하기
            last_10_days = scaled_data[-10:].reshape(1, 10, 1)

            # 5. 모델 불러오기 및 예측
            # (주의: 미리 학습된 모델이 models 폴더에 있어야 합니다)
            try:
                model = load_model("models/my_stock_model.h5")
                prediction = model.predict(last_10_days)
                predicted_price = scaler.inverse_transform(prediction)

                # 6. 결과 보여주기
                today_price = data[-1][0]
                pred_price = predicted_price[0][0]

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="오늘의 종가", value=f"{today_price:,.0f}원")
                with col2:
                    diff = pred_price - today_price
                    st.metric(
                        label="내일 예측가",
                        value=f"{pred_price:,.0f}원",
                        delta=f"{diff:,.0f}원",
                    )

                # 7. 차트 그리기
                st.subheader("📊 최근 주가 흐름")
                chart_data = df[["Close"]].tail(30)
                st.line_chart(chart_data)

            except Exception as e:
                st.error(
                    f"모델을 불러오는 데 실패했습니다. 먼저 '02_모델학습.ipynb'를 실행해서 모델을 만들어주세요! 에러: {e}"
                )

# 사이드바 설명
with st.sidebar:
    st.header("사용 가이드")
    st.write("1. **종목 코드**를 입력하세요.")
    st.write("2. **예측 시작** 버튼을 누르세요.")
    st.write("3. AI가 분석한 결과를 확인하세요!")
    st.info("이 예측은 재미로만 봐주세요. 투자의 책임은 본인에게 있습니다. 😂")
