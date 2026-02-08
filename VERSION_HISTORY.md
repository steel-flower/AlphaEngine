# Alpha Engine Sigma - Version History

## 버전 관리 정책
- 각 주요 버전은 `old_versions/` 폴더에 백업됩니다
- 파일명 형식: `alpha_engine_sigma_v{버전}.py`
- 심각한 에러 발생 시 이전 버전으로 복원 가능

---

## v7.2 (2026-02-05) - Win-Rate Priority Optimization
**목표**: 적중률 60% 이상 달성

### 주요 변경사항
1. **적중률 우선 최적화**
   - 적중률 60% 목표 설정
   - Win-rate bonus: (avg_win_rate - 0.6) × 15,000점
   
2. **파라미터 그리드 확장**
   - `entry_threshold`: 0.35~0.65 (더 높은 문턱)
   - `tp_atr`: 3.0~6.0 (더 긴 익절)
   - `sl_atr`: 1.2~2.0 (더 긴 손절)
   - `lrs_period`: 30~75 (더 긴 추세)
   
3. **학습 강화**
   - Epochs: 15 → 20
   - 탐색 조합: 12 → 20개

4. **리포트 개선**
   - 평균 보유일 추가
   - 표 정렬 개선 (85자 너비)

### 복원 방법
```batch
copy old_versions\alpha_engine_sigma_v7.2.py alpha_engine_sigma.py
```

---

## v7.1 (2026-02-05) - 10% Annual Target
**목표**: 연간 최소 10% 수익률 달성

### 주요 변경사항
1. **연도별 평균 보유일 추가**
2. **10% 미달 연도 페널티 강화**: -5,000점
3. **파라미터 그리드 확장**
   - `lrs_period`: 30, 45, 60
   - `entry_threshold`: 0.25, 0.35, 0.45
   - `tp_atr`: 2.5, 3.2, 4.0
   - `sl_atr`: 1.0, 1.2, 1.5
4. **학습 강화**: Epochs 10 → 15
5. **탐색 조합**: 5 → 12개

### 복원 방법
```batch
copy old_versions\alpha_engine_sigma_v7.1.py alpha_engine_sigma.py
```

---

## v7.0 (2026-02-05) - Sacred Restoration
**목표**: 기초 코드 100% 복원

### 주요 변경사항
1. **기초 코드 완전 복원**
   - TransformerPredictor (LSTM이 아닌 Transformer)
   - 17개 피처 전체 복원
   - Tech_Score 4중 합산 (DMI + OBV + MACD + Stoch)
   
2. **최적화 함수 복원**
   - Stability-First Search
   - 손실 연도 페널티: -5,000점
   - MDD 페널티: -2,000점

3. **트레이딩 로직 복원**
   - YTD 동적 조정
   - VIX 리바운드 감지
   - 30일 보유 제한
   - 켈리 공식 포지션 사이징

### 복원 방법
```batch
copy old_versions\alpha_engine_sigma_v7.0.py alpha_engine_sigma.py
```

---

## v6.0 (2026-02-05) - Pure Edition (실패)
**문제**: AI 지휘자 개념 제거 시도 중 핵심 로직 훼손
- 수익률: 900%+ → 37% 폭락
- 원인: Tech_Score 삭제, LSTM 사용, 6개 피처만 사용

**교훈**: 기초 코드의 핵심 요소를 임의로 제거하지 말 것

---

## 이전 버전들
- v5.0: Triple Pillar (LRS, ML RSI, LSTM) - 망상적 개념
- v4.0: Pure Mathematical - AI 제거 시도
- v3.0: Sacred Logic 복원 시도
- v2.2: Listing Date Perfect - 데이터 무결성 확보
- v2.1: Error Fixed - 리포팅 버그 수정
- v2.0: True Evolution - 최적화 로직 수정
- v1.9: Stable Version - 시드 고정
- v1.8: Sell-only cost (0.3%)
- v1.7: Win-rate 추가
- v1.6: Kelly Criterion 복원
- v1.5: 초기 버전

---

## 긴급 복원 가이드

### 1. 최신 안정 버전으로 복원
```batch
cd C:\Users\user\Desktop\Antigravity\AlphaEngine
copy old_versions\alpha_engine_sigma_v7.2.py alpha_engine_sigma.py
```

### 2. 특정 버전으로 복원
```batch
copy old_versions\alpha_engine_sigma_v7.0.py alpha_engine_sigma.py
```

### 3. 복원 후 확인
```batch
run_SIGMA.bat
```

---

## 버전 백업 체크리스트
- [ ] `old_versions/` 폴더에 백업 완료
- [ ] VERSION_HISTORY.md 업데이트
- [ ] 주요 변경사항 문서화
- [ ] 복원 방법 명시
