import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from alpha_engine_sigma import AlphaEngineSigma

def force_sync():
    print("\n" + "="*70)
    print("    [v3.4.5] SYSTEM RESET & HIGH-PRECISION SYNC")
    print("    Deleting old data and performing fresh analysis...")
    print("="*70 + "\n")

    # 1. Delete old data
    if os.path.exists("dashboard_data.json"):
        os.remove("dashboard_data.json")
        print(" [!] Deleted old dashboard_data.json")

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
            print(f" [Data] Analyzing {name} ({ticker})...")
            engine = AlphaEngineSigma(ticker, name)
            engine.fetch_data()
            
            # Legacy Params
            pattern = os.path.join(engine.log_dir, f"{name}_{ticker}_*.txt")
            all_logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            
            if not all_logs:
                engine.best_params = {'entry_threshold': 0.35, 'tp_atr': 3.2, 'sl_atr': 1.5}
            else:
                with open(all_logs[0], 'r', encoding='utf-8') as f:
                    meta = json.loads(f.read().split("### METADATA ###")[1].strip())
                    engine.best_params = meta['params']
            
            # Fast Train
            split = int(len(engine.df)*0.8)
            df_tr = engine.add_indicators(engine.df.iloc[:split])
            Xt, yt = engine.prepare_sequences(df_tr)
            engine.model = engine.train(Xt, yt, 42)
            
            # Evaluate
            df_te = engine.add_indicators(engine.df)
            res, met = engine.evaluate_strategy(df_te, engine.best_params)
            
            last = res.iloc[-1]
            lp = float(last['Close'])
            score = float(last['Total_Score'])
            latr = float(last['ATR'])
            
            tp_m = float(engine.best_params.get('tp_atr', 3.2))
            sl_m = float(engine.best_params.get('sl_atr', 1.5))
            if engine.is_downtrend_mode: tp_m *= 0.6; sl_m *= 1.2
            
            current_signal = "wait"
            eth = float(engine.best_params.get('entry_threshold', 0.35))
            pot_profit = (tp_m * latr) / (lp + 1e-9)
            
            if score > eth or (score > 0 and (last['Disp_20'] < 0.94 or last['RSI'] < 24)):
                if pot_profit >= 0.05:
                    current_signal = "buy"

            # Create Hist (ULTRA EXPLICIT)
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
            print(f"  - OK: Price={lp}, Hist_Len={len(hist)}")
            
        except Exception as e:
            print(f" [!] Error {name}: {e}")

    # 2. Save and Sync
    with open("dashboard_data.json", "w", encoding='utf-8') as f:
        json.dump(dashboard_results, f, indent=4, ensure_ascii=False)
    
    print("\n [Sync] Pushing to GitHub...")
    os.system("git add dashboard_data.json")
    os.system('git commit -m "[v3.4.5] Fresh Data Reset"')
    os.system("git push origin main --force")
    print("\n [Done] Check web app in 1 minute.")

if __name__ == "__main__":
    force_sync()
