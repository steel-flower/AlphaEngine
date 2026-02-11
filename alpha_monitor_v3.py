import time
import os
import json
from datetime import datetime
from alpha_engine_sigma import AlphaEngineSigma
from email_notifier import EmailNotifier
import warnings
import glob

warnings.filterwarnings('ignore')

def is_market_open():
    """한국 시장 개장 시간 체크 (09:00 ~ 15:30)"""
    now = datetime.now()
    # 주말 체크
    if now.weekday() >= 5:
        return False
    
    start_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return start_time <= now <= end_time

def run_monitor():
    notifier = EmailNotifier()
    assets = [
        ("KODEX 코스피", "226490.KS"), 
        ("KODEX 인버스", "114800.KS"), 
        ("ACE KRX 금현물", "411060.KS"), 
        ("KODEX 은선물(H)", "144600.KS"), 
        ("RISE 글로벌 자산배분 액티브", "461490.KS"), 
        ("RISE 글로벌농업경제", "437370.KS"), 
        ("KODEX WTI 원유선물(H)", "261220.KS"), 
        ("삼성전자", "005930.KS"), 
        ("SK 하이닉스", "000660.KS")
    ]
    
    # 마지막으로 보낸 신호를 저장하여 중복 발송 방지
    # {ticker: "buy" | "sell" | "wait"}
    last_signals = {}
    
    print("\n" + "="*70)
    print("    ALPHA ENGINE SIGMA v3.2 REAL-TIME MONITOR")
    print("    Monitoring starting... (Interval: 5 Minutes)")
    print("    Recipient: frederic.jeon@gmail.com")
    print("="*70 + "\n")

    while True:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if not is_market_open():
            # 장 종료 후 또는 주말에는 30분 단위로 체크
            print(f" [{now_str}] 장외 시간입니다. 다음 체크까지 대기합니다...")
            time.sleep(1800)
            continue

        print(f"\n [{now_str}] 실시간 시장 분석 시작...")
        dashboard_results = []
        
        for name, ticker in assets:
            try:
                engine = AlphaEngineSigma(ticker, name)
                engine.fetch_data()
                
                # [STRATEGIC FIX] Skip evolution. Strictly inherit latest legacy intelligence.
                # Find the best log for this ticker to get params
                pattern = os.path.join(engine.log_dir, f"{name}_{ticker}_*.txt")
                all_logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                
                if not all_logs:
                    print(f" [!] {name}: 최적화 로그를 찾을 수 없습니다. Sigma를 먼저 실행하세요.")
                    continue
                
                with open(all_logs[0], 'r', encoding='utf-8') as f:
                    meta = json.loads(f.read().split("### METADATA ###")[1].strip())
                    engine.best_params = meta['params']
                    # Re-train model only ONCE with the inherited params to get the inference capability
                    # (Weights are not saved to disk, so we need one fast training to active the brain)
                    split = int(len(engine.df)*0.8)
                    df_tr = engine.add_indicators(engine.df.iloc[:split])
                    Xt, yt = engine.prepare_sequences(df_tr)
                    engine.model = engine.train(Xt, yt, 42) # Single seed pass for speed
                
                # Real-time Signal Detection
                df_te = engine.add_indicators(engine.df)
                res, met = engine.evaluate_strategy(df_te, engine.best_params)
                
                last = res.iloc[-1]
                lp = last['Close']
                score = last['Total_Score']
                # Re-calculate potential profit for 5% guard
                latr = last['ATR']
                tp_m = engine.best_params.get('tp_atr', 3.2)
                sl_m = engine.best_params.get('sl_atr', 1.5)
                if engine.is_downtrend_mode:
                    tp_m *= 0.6
                    sl_m *= 1.2
                potential_profit = (tp_m * latr) / (lp + 1e-9)
                
                eth = engine.best_params.get('entry_threshold', 0.35)
                current_signal = "wait"
                if score > eth or (score > 0 and (last['Disp_20'] < 0.94 or last['RSI'] < 24)):
                    if potential_profit >= 0.05:
                        current_signal = "buy"
                
                # 신호 변화 감지 시 이메일 발송
                if ticker not in last_signals or last_signals[ticker] != current_signal:
                    if current_signal == "buy":
                        entry_p = lp
                        target_p = lp + tp_m * latr
                        sl_p = lp - sl_m * latr
                        
                        notifier.send_buy_signal(
                            ticker=ticker,
                            name=name,
                            current_price=lp,
                            entry_price=entry_p,
                            target_price=target_p,
                            stop_loss=sl_p,
                            ai_score=score,
                            tech_score=0 # v3.2는 AI Score에 통합됨
                        )
                        print(f" [Signal] {name}: 매수 신호 발생! 이메일 발송 완료.")
                    
                    elif current_signal == "wait" and last_signals.get(ticker) == "buy":
                        # 매수 상태에서 관망으로 변했다는 것은 청산 신호로 간주
                        notifier.send_sell_signal(
                            ticker=ticker,
                            name=name,
                            entry_price=lp, # 실제 진입가는 모르므로 현재가로 대체
                            current_price=lp,
                            reason="AI Score 하락 또는 기대수익 미달 (관망 전환)"
                        )
                        print(f" [Signal] {name}: 포지션 종료/관망 전환. 이메일 발송 완료.")
                    
                    last_signals[ticker] = current_signal
                
                print(f"  - {name:25s}: Score {score:6.2f} | Status: {current_signal.upper()}")
                
                # [v3.4 Bridge] Prepare data for Web Dashboard
                dashboard_results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": lp,
                    "score": score,
                    "signal": current_signal,
                    "potential_profit": potential_profit * 100,
                    "entry_price": engine.best_params.get('entry_threshold_px', lp), # Simplified for dashboard
                    "target_price": lp + (tp_m * latr),
                    "stop_loss": lp - (sl_m * latr),
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "history": res[['Close', 'Total_Score', 'Signal']].tail(120).reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records')
                })
                
            except Exception as e:
                print(f" [!] {name} 분석 중 오류 발생: {e}")
        
        # Save shared data for Streamlit
        try:
            with open("dashboard_data.json", "w", encoding='utf-8') as f:
                json.dump(dashboard_results, f, indent=4, ensure_ascii=False)
            
            # [v3.4 Auto-Sync] Push to GitHub for live web dashboard
            print("\n [v3.4] GitHub 저장소로 데이터를 전송합니다...")
            os.system("git add dashboard_data.json")
            os.system('git commit -m "Manual Force Sync for Dashbord Test"')
            os.system("git push origin main --force")
            print("\n 완료! 1분 뒤 웹페이지를 새로고침 하세요.")
            
        except Exception as e:
            print(f" [!] 대시보드 데이터 저장/동기화 오류: {e}")
        
        print(f"\n [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 분석 완료. 5분 뒤 다시 시작합니다.")
        time.sleep(300) # 5분 대기

if __name__ == "__main__":
    run_monitor()
