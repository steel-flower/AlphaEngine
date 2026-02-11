import os
import json
from datetime import datetime
from alpha_monitor_v3 import AlphaEngineSigma, EmailNotifier
import glob

def force_sync():
    print("\n" + "="*70)
    print("    [v3.4] 웹 대시보드 강제 데이터 동기화 (Test Mode)")
    print("    시간 체크를 무시하고 현재 시점의 데이터를 GitHub로 전송합니다.")
    print("="*70 + "\n")

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
    
    dashboard_results = []
    
    for name, ticker in assets:
        try:
            print(f" [Data] Fetching {name} ({ticker})...")
            engine = AlphaEngineSigma(ticker, name)
            engine.fetch_data()
            
            # Legacy 로그에서 최적 파라미터 수입
            pattern = os.path.join(engine.log_dir, f"{name}_{ticker}_*.txt")
            all_logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            
            if not all_logs:
                print(f" [!] {name}: 로그 없음. 기본값 분석.")
                engine.best_params = {'entry_threshold': 0.35, 'tp_atr': 3.2, 'sl_atr': 1.5}
            else:
                with open(all_logs[0], 'r', encoding='utf-8') as f:
                    meta = json.loads(f.read().split("### METADATA ###")[1].strip())
                    engine.best_params = meta['params']
            
            # 속성 모델 훈련 (Single Pass)
            split = int(len(engine.df)*0.8)
            df_tr = engine.add_indicators(engine.df.iloc[:split])
            Xt, yt = engine.prepare_sequences(df_tr)
            engine.model = engine.train(Xt, yt, 42)
            
            # 전략 평가
            df_te = engine.add_indicators(engine.df)
            res, met = engine.evaluate_strategy(df_te, engine.best_params)
            
            last = res.iloc[-1]
            lp = last['Close']
            score = last['Total_Score']
            latr = last['ATR']
            tp_m = engine.best_params.get('tp_atr', 3.2)
            sl_m = engine.best_params.get('sl_atr', 1.5)
            if engine.is_downtrend_mode: tp_m *= 0.6; sl_m *= 1.2
            
            potential_profit = (tp_m * latr) / (lp + 1e-9)
            eth = engine.best_params.get('entry_threshold', 0.35)
            
            current_signal = "wait"
            if score > eth or (score > 0 and (last['Disp_20'] < 0.94 or last['RSI'] < 24)):
                if potential_profit >= 0.05:
                    current_signal = "buy"

            print(f"  - {name:25s}: Score {score:6.2f} | Status: {current_signal.upper()}")
            
            dashboard_results.append({
                "ticker": ticker,
                "name": name,
                "price": lp,
                "score": score,
                "signal": current_signal,
                "potential_profit": potential_profit * 100,
                "entry_price": engine.best_params.get('entry_threshold_px', lp),
                "target_price": lp + (tp_m * latr),
                "stop_loss": lp - (sl_m * latr),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "history": res[['Close', 'Total_Score', 'Signal']].tail(120).reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records')
            })
        except Exception as e:
            print(f" [!] {name} 오류: {e}")

    # 파일 저장 및 GitHub 전송
    with open("dashboard_data.json", "w", encoding='utf-8') as f:
        json.dump(dashboard_results, f, indent=4, ensure_ascii=False)
    
    print("\n [v3.4] GitHub 저장소로 데이터를 전송합니다...")
    os.system("git add dashboard_data.json")
    os.system('git commit -m "Manual Force Sync for Dashbord Test"')
    os.system("git push origin main --force")
    print("\n 완료! 1분 뒤 웹페이지를 새로고침 하세요.")

if __name__ == "__main__":
    force_sync()
