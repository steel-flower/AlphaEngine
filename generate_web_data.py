from alpha_engine_sigma import AlphaEngineSigma
import json
import os

def generate_all():
    if not os.path.exists("assets.json"):
        print("assets.json 파일이 없습니다.")
        return
        
    with open("assets.json", "r", encoding='utf-8') as f:
        assets = json.load(f)
        
    for a in assets:
        print(f"\n>>> {a['name']} ({a['ticker']}) 데이터 생성 중...")
        try:
            engine = AlphaEngineSigma(a['ticker'], a['name'])
            engine.fetch_data()
            
            # 빠른 생성을 위해 epochs=5로 학습
            f_df = engine.add_indicators(engine.df)
            Xt, yt = engine.prepare_sequences(f_df)
            
            # 학습 (이미 최적화된 파라미터가 있다면 그것을 사용)
            engine.model = engine.train(Xt, yt, epochs=5)
            
            # 성과 평가 및 웹 데이터 저장
            res_df, met = engine.evaluate_strategy(f_df)
            engine.save_web_data(res_df, met)
            print(f"✅ {a['name']} 완료")
        except Exception as e:
            print(f"❌ {a['name']} 실패: {e}")

if __name__ == "__main__":
    generate_all()
