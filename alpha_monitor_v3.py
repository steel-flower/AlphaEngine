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
    if now.weekday() >= 5: return False
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
    
    last_signals = {}
    
    print("\n" + "="*70)
    print("    ALPHA ENGINE SIGMA v3.4.5 REAL-TIME MONITOR")
    print("    Monitoring starting... (Interval: 5 Minutes)")
    print("="*70 + "\n")

    while True:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if not is_market_open():
            print(f" [{now_str}] 장외 시간입니다. 30분 대기...")
            time.sleep(1800)
            continue

        print(f"\n [{now_str}] 실시간 시장 분석 시작...")
        dashboard_results = []
        
        for name, ticker in assets:
            try:
                engine = AlphaEngineSigma(ticker, name)
                engine.fetch_data()
                
                # Load Best Params from Logs
                pattern = os.path.join(engine.log_dir, f"{name}_{ticker}_*.txt")
                all_logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                
                if not all_logs:
                    engine.best_params = {'entry_threshold': 0.35, 'tp_atr': 3.2, 'sl_atr': 1.5}
                else:
                    with open(all_logs[0], 'r', encoding='utf-8') as f:
                        meta = json.loads(f.read().split("### METADATA ###")[1].strip())
                        engine.best_params = meta['params']
                
                # Single Seed Train for Inference
                split = int(len(engine.df)*0.8)
                df_tr = engine.add_indicators(engine.df.iloc[:split])
                Xt, yt = engine.prepare_sequences(df_tr)
                engine.model = engine.train(Xt, yt, 42)
                
                # Signal Detection
                df_te = engine.add_indicators(engine.df)
                res, met = engine.evaluate_strategy(df_te, engine.best_params)
                
                last = res.iloc[-1]
                lp = float(last['Close'])
                score = float(last['Total_Score'])
                latr = float(last['ATR'])
                
                tp_m = float(engine.best_params.get('tp_atr', 3.2))
                sl_m = float(engine.best_params.get('sl_atr', 1.5))
                if engine.is_downtrend_mode: tp_m *= 0.6; sl_m *= 1.2
                
                pot_profit = (tp_m * latr) / (lp + 1e-9)
                eth = float(engine.best_params.get('entry_threshold', 0.35))
                
                current_signal = "wait"
                if score > eth or (score > 0 and (last['Disp_20'] < 0.94 or last['RSI'] < 24)):
                    if pot_profit >= 0.05:
                        current_signal = "buy"
                
                # Email Notification
                if ticker not in last_signals or last_signals[ticker] != current_signal:
                    if current_signal == "buy":
                        notifier.send_buy_signal(ticker, name, lp, lp, lp + tp_m*latr, lp - sl_m*latr, score, 0)
                    elif current_signal == "wait" and last_signals.get(ticker) == "buy":
                        notifier.send_sell_signal(ticker, name, lp, lp, "AI Score 하락 또는 기대수익 미달")
                    last_signals[ticker] = current_signal
                
                # History Collection (Robust)
                hist = []
                tail = res.tail(120)
                for d, r in tail.iterrows():
                    hist.append({
                        "Date": d.strftime('%Y-%m-%d'),
                        "Close": round(float(r['Close']), 2),
                        "Score": round(float(r['Total_Score']), 4),
                        "Sig": int(r['Signal'])
                    })

                dashboard_results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": lp,
                    "score": score,
                    "signal": current_signal,
                    "pot": pot_profit * 100,
                    "target": lp + (tp_m * latr),
                    "stop": lp - (sl_m * latr),
                    "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "history": hist
                })
                print(f"  - {name:25s}: Score {score:6.2f} | Status: {current_signal.upper()}")
                
            except Exception as e:
                print(f" [!] {name} 분석 중 오류 발생: {e}")
        
        # Save and Sync
        try:
            with open("dashboard_data.json", "w", encoding='utf-8') as f:
                json.dump(dashboard_results, f, indent=4, ensure_ascii=False)
            
            print("\n [v3.4.5] GitHub 데이터 동기화 시작...")
            os.system("git add dashboard_data.json")
            os.system('git commit -m "Auto Monitor Sync v3.4.5"')
            os.system("git push origin main --force")
            print(" 완료!")
            
        except Exception as e:
            print(f" [!] 데이터 동기화 오류: {e}")
        
        time.sleep(300)

if __name__ == "__main__":
    run_monitor()
