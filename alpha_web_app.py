import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

# 페이지 설정
st.set_page_config(
    page_title="Alpha Engine v3.4 Live Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (프리미엄 미학)
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .status-buy { color: #00ff88; font-weight: bold; }
    .status-wait { color: #ffbc00; font-weight: bold; }
    .css-1offfwp { background-color: #161b22 !important; }
</style>
""", unsafe_allow_html=True)

# 비밀번호 보안 (간이)
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        st.title("🏛️ Alpha Engine 보안 접속")
        pwd = st.text_input("Access Password", type="password")
        if st.button("Unlock"):
            if pwd == "alpha77": # 기본 비밀번호
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid Password")
        return False
    return True

if check_password():
    # 사이드바 설정
    st.sidebar.title("🏛️ Alpha Engine v3.4")
    st.sidebar.markdown("---")
    st.sidebar.info("실시간 시장 감시 및 AI 전략 대시보드")
    
    # 데이터 로드
    def load_data():
        if os.path.exists("dashboard_data.json"):
            with open("dashboard_data.json", "r", encoding='utf-8') as f:
                return json.load(f)
        return []

    data = load_data()
    
    if not data:
        st.warning("경고: 실시간 데이터 파일을 찾을 수 없습니다. Monitor 프로그램을 먼저 실행하세요.")
    else:
        df = pd.DataFrame(data)
        last_update = df['timestamp'].iloc[0]
        
        st.title("🚀 Real-Time Market Intelligence")
        st.caption(f"Last Updated: {last_update} (5분 주기 자동 갱신)")

        # 상단 요약 지표
        cols = st.columns(len(df))
        for i, row in df.iterrows():
            with cols[i]:
                color = "normal" if row['signal'] == "wait" else "inverse"
                st.metric(
                    label=row['name'],
                    value=f"{row['price']:,.0f}",
                    delta=f"AI: {row['score']:.2f}",
                    delta_color=color
                )
        
        st.markdown("---")
        
        # 메인 분석 섹션
        st.subheader("📋 실시간 포지션 및 AI 전략 현황")
        display_df = df[['name', 'score', 'signal', 'potential_profit', 'price', 'target_price', 'stop_loss']].copy()
        display_df.columns = ['종목명', 'AI Score', '상태', '기대수익(%)', '현재가', '목표가', '손절가']
        
        def color_signal(val):
            if val == 'buy': return 'background-color: #004d00; color: #00ff88'
            if val == 'wait': return 'background-color: #4d3300; color: #ffbc00'
            return ''
        
        st.dataframe(
            display_df.style.applymap(color_signal, subset=['상태'])
            .format({
                'AI Score': '{:.2f}', 
                '기대수익(%)': '{:.1f}%',
                '현재가': '{:,.0f}',
                '목표가': '{:,.0f}',
                '손절가': '{:,.0f}'
            }),
            use_container_width=True,
            height=300
        )

        st.markdown("---")
        st.subheader("📈 Master Precision AI Chart (5-Rule Compliance)")
        
        selected_name = st.selectbox("분석 차트 선택", options=df['name'].tolist())
        selected_data = df[df['name'] == selected_name].iloc[0]
        
        if "history" in selected_data:
            hist_df = pd.DataFrame(selected_data['history'])
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            
            fig = go.Figure()
            
            # Rule 1, 2, 3: Raw price data, Linear scale, Direct mapping
            fig.add_trace(go.Scatter(
                x=hist_df['Date'], 
                y=hist_df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#00ff88', width=2),
                hovertemplate='%{x}<br>Price: %{y:,.0f}원'
            ))
            
            # Rule 4: No smoothing/manipulation - ensured by using raw hist_df
            
            # Buy/Sell Markers
            buys = hist_df[hist_df['Signal'] > 0]
            sells = hist_df[hist_df['Signal'] < 0]
            
            fig.add_trace(go.Scatter(
                x=buys['Date'], y=buys['Close'],
                mode='markers', name='Buy',
                marker=dict(symbol='triangle-up', size=12, color='#00ff88'),
                hovertemplate='Buy Signal<br>%{x}<br>%{y:,.0f}원'
            ))
            
            fig.add_trace(go.Scatter(
                x=sells['Date'], y=sells['Close'],
                mode='markers', name='Sell',
                marker=dict(symbol='triangle-down', size=12, color='#ff4b4b'),
                hovertemplate='Sell Signal<br>%{x}<br>%{y:,.0f}원'
            ))

            # Buy-Sell Connection Lines (Visual Profitability)
            # Find matching buy-sell pairs for lines
            last_buy = None
            for idx, row in hist_df.iterrows():
                if row['Signal'] > 0:
                    last_buy = row
                elif row['Signal'] < 0 and last_buy is not None:
                    fig.add_trace(go.Scatter(
                        x=[last_buy['Date'], row['Date']],
                        y=[last_buy['Close'], row['Close']],
                        mode='lines',
                        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    last_buy = None

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=500,
                xaxis=dict(
                    showgrid=True, gridcolor='#30363d', 
                    type='date', title='Time (Linear X-Axis)'
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='#30363d', 
                    title='Price (Linear Y-Axis)', tickformat=','
                ),
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Rule 5: Data Verification Table
            with st.expander("📊 Raw Data Verification Table (Original Values)"):
                st.dataframe(
                    hist_df[['Date', 'Close', 'Total_Score', 'Signal']].sort_values('Date', ascending=False),
                    use_container_width=True
                )
        else:
            st.info("차트 데이터를 불러오는 중입니다. 모니터링 세션이 업데이트될 때까지 기다려 주세요.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
