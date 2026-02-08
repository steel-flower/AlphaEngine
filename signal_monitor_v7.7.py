"""
Alpha Engine v7.7 - 신호 모니터링 시스템
카카오톡 알림과 함께 실시간 매매 신호 감지
"""
import time
import schedule
from datetime import datetime, time as dt_time
import sys
import os

# Alpha Engine 모듈 import
import sys
sys.path.append('.')
from alpha_engine_sigma import AlphaEngineSigma
import json

def load_assets():
    """assets.json에서 종목 리스트 로드"""
    with open('assets.json', 'r', encoding='utf-8') as f:
        return json.load(f)

from email_notifier import EmailNotifier


class SignalMonitor:
    def __init__(self):
        self.notifier = EmailNotifier()
        self.assets = load_assets()
        self.positions = {}  # {ticker: {'entry_price': float, 'entry_date': str}}
        self.last_signals = {}  # {ticker: 'buy'/'sell'/'hold'}
        
    def check_market_hours(self):
        """한국 주식시장 운영 시간 확인"""
        now = datetime.now()
        current_time = now.time()
        
        # 주말 제외
        if now.weekday() >= 5:  # 토요일(5), 일요일(6)
            return False
        
        # 장 시작 전: 08:30 - 09:00
        pre_market = dt_time(8, 30) <= current_time < dt_time(9, 0)
        
        # 정규 장: 09:00 - 15:30
        regular_market = dt_time(9, 0) <= current_time < dt_time(15, 30)
        
        return pre_market or regular_market
    
    def analyze_all_assets(self):
        """모든 종목 분석 및 신호 감지"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 전체 종목 분석 시작...")
        
        for asset in self.assets:
            try:
                self.analyze_asset(asset)
            except Exception as e:
                print(f"[오류] {asset['name']} 분석 실패: {e}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 분석 완료\n")
    
    def analyze_asset(self, asset):
        """개별 종목 분석"""
        ticker = asset['ticker']
        name = asset['name']
        
        print(f"  - {name} ({ticker}) 분석 중...")
        
        # Alpha Engine 실행
        engine = AlphaEngineSigma(ticker=ticker, name=name)
        
        try:
            # 데이터 수집 및 분석 (최적화는 스킵)
            df = engine.fetch_data()
            if df is None or len(df) < 100:
                print(f"    [경고] {name}: 데이터 부족")
                return
            
            # 특징 생성
            df = engine.build_features(df)
            
            # AI 모델 학습 (간단히)
            engine.train(df)
            
            # 전략 평가
            res_df, met = engine.evaluate_strategy(df)
            
            # 최신 신호 확인
            latest = res_df.iloc[-1]
            current_price = latest['Close']
            ai_score = latest['AI_Score']
            tech_score = latest['Tech_Score']
            
            # 진입 조건 확인
            entry_threshold = engine.best_params['entry_threshold']
            is_buy_signal = ai_score > entry_threshold and tech_score > 0.3
            
            # 포지션 확인
            has_position = ticker in self.positions
            
            # 매수 신호
            if is_buy_signal and not has_position:
                if self.last_signals.get(ticker) != 'buy':
                    self.send_buy_alert(ticker, name, current_price, latest, engine.best_params)
                    self.last_signals[ticker] = 'buy'
            
            # 청산 신호 (포지션 있을 때)
            elif has_position:
                entry_price = self.positions[ticker]['entry_price']
                atr = latest['ATR']
                
                tp = entry_price + engine.best_params['tp_atr'] * atr
                sl = entry_price - engine.best_params['sl_atr'] * atr
                
                # 익절 또는 손절
                if current_price >= tp:
                    self.send_sell_alert(ticker, name, entry_price, current_price, "목표가 도달")
                    del self.positions[ticker]
                    self.last_signals[ticker] = 'sell'
                
                elif current_price <= sl:
                    self.send_sell_alert(ticker, name, entry_price, current_price, "손절가 도달")
                    del self.positions[ticker]
                    self.last_signals[ticker] = 'sell'
                
                elif ai_score < -0.4:
                    self.send_sell_alert(ticker, name, entry_price, current_price, "신호 약화")
                    del self.positions[ticker]
                    self.last_signals[ticker] = 'sell'
        
        except Exception as e:
            print(f"    [오류] {name} 분석 중 에러: {e}")
    
    def send_buy_alert(self, ticker, name, current_price, latest, params):
        """매수 알림 전송"""
        atr = latest['ATR']
        entry_price = current_price
        target_price = entry_price + params['tp_atr'] * atr
        stop_loss = entry_price - params['sl_atr'] * atr
        ai_score = latest['AI_Score']
        tech_score = latest['Tech_Score']
        
        print(f"    [매수 신호] {name}: {current_price:,.0f}원")
        
        # 카카오톡 알림
        self.notifier.send_buy_signal(
            ticker, name, current_price, entry_price,
            target_price, stop_loss, ai_score, tech_score
        )
        
        # 포지션 기록 (사용자가 실제로 매수했다고 가정)
        self.positions[ticker] = {
            'entry_price': entry_price,
            'entry_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def send_sell_alert(self, ticker, name, entry_price, current_price, reason):
        """청산 알림 전송"""
        profit_pct = (current_price / entry_price - 1) * 100
        print(f"    [청산 신호] {name}: {profit_pct:+.2f}% ({reason})")
        
        # 카카오톡 알림
        self.notifier.send_sell_signal(ticker, name, entry_price, current_price, reason)
    
    def morning_analysis(self):
        """장 시작 전 분석"""
        print("\n" + "="*60)
        print(f"[장 시작 전 분석] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        self.analyze_all_assets()
        
        # 일일 요약 전송
        summary = f"""오늘의 분석이 완료되었습니다.

분석 종목: {len(self.assets)}개
보유 포지션: {len(self.positions)}개

매수 신호가 발생하면 알림을 보내드립니다."""
        
        self.notifier.send_daily_summary(summary)
    
    def intraday_check(self):
        """장 중 체크"""
        if not self.check_market_hours():
            return
        
        print(f"\n[장 중 체크] {datetime.now().strftime('%H:%M:%S')}")
        
        # 포지션이 있는 종목만 체크
        if self.positions:
            for asset in self.assets:
                if asset['ticker'] in self.positions:
                    self.analyze_asset(asset)
        else:
            print("  보유 포지션 없음")
    
    def run(self):
        """모니터링 시작"""
        print("\n" + "="*60)
        print("Alpha Engine v7.7 - 신호 모니터링 시작")
        print("="*60)
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"모니터링 종목: {len(self.assets)}개")
        print("="*60 + "\n")
        
        # 스케줄 설정
        schedule.every().day.at("08:30").do(self.morning_analysis)  # 장 시작 전
        schedule.every(5).minutes.do(self.intraday_check)  # 5분마다 체크
        
        # 즉시 한 번 실행
        self.morning_analysis()
        
        # 무한 루프
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # 30초마다 스케줄 체크
        except KeyboardInterrupt:
            print("\n\n모니터링 종료")


if __name__ == "__main__":
    monitor = SignalMonitor()
    monitor.run()
