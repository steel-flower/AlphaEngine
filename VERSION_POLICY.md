# Alpha Engine Sigma - 버전 관리 원칙

## 📋 필수 원칙

### 1. 모든 버전은 old_versions/ 폴더에 백업
- **위치**: `C:\Users\user\Desktop\Antigravity\AlphaEngine\old_versions\`
- **파일명 형식**: `alpha_engine_sigma_v{버전}.py`
- **예시**: `alpha_engine_sigma_v7.3.py`

### 2. 버전 업그레이드 시 필수 작업
```powershell
# 1단계: 현재 버전을 old_versions에 백업
Copy-Item -Path alpha_engine_sigma.py -Destination old_versions\alpha_engine_sigma_v{현재버전}.py -Force

# 2단계: 코드 수정 진행

# 3단계: 수정 완료 후 다시 백업
Copy-Item -Path alpha_engine_sigma.py -Destination old_versions\alpha_engine_sigma_v{새버전}.py -Force
```

### 3. 버전 표기 통일
모든 버전 표시는 **일관되게** 유지:
- 배치 파일 (`run_SIGMA.bat`)
- 메뉴 타이틀 (`main()` 함수)
- 최적화 메시지 (`run_optimization()`)
- 최종 리포트 (`generate_report()`)

---

## 📁 현재 백업 상태

### old_versions/ 폴더
```
old_versions/
├── alpha_engine_sigma_v7.2.py  (2026-02-05 23:41)
└── alpha_engine_sigma_v7.3.py  (2026-02-06 15:48)
```

---

## 🔄 버전 복원 방법

### 특정 버전으로 복원
```powershell
Copy-Item -Path old_versions\alpha_engine_sigma_v7.3.py -Destination alpha_engine_sigma.py -Force
```

### 또는 복원 스크립트 사용
```batch
restore_version.bat v7.3
```

---

## ✅ 버전 업그레이드 체크리스트

매번 버전을 올릴 때마다 확인:

- [ ] **1단계**: 현재 버전을 `old_versions/`에 백업
- [ ] **2단계**: 코드 수정 (기능 추가/버그 수정)
- [ ] **3단계**: 모든 버전 표기 통일 (배치/메뉴/리포트)
- [ ] **4단계**: 수정된 버전을 `old_versions/`에 백업
- [ ] **5단계**: `VERSION_HISTORY.md` 업데이트
- [ ] **6단계**: 테스트 실행 (`run_SIGMA.bat`)

---

## 📝 버전 명명 규칙

### 메이저 버전 (v7.0, v8.0)
- 핵심 로직 변경
- 아키텍처 재설계
- 호환성 깨짐

### 마이너 버전 (v7.1, v7.2, v7.3)
- 기능 추가
- 최적화 개선
- 버그 수정

### 서브 타이틀
- v7.0 (SACRED RESTORATION)
- v7.1 (10% Annual Target)
- v7.2 (Win-Rate Priority)
- v7.3 (DETERMINISTIC)

---

## 🚨 긴급 복원 시나리오

### 심각한 버그 발생 시
```powershell
# 1. 즉시 이전 안정 버전으로 복원
Copy-Item -Path old_versions\alpha_engine_sigma_v7.2.py -Destination alpha_engine_sigma.py -Force

# 2. 테스트
run_SIGMA.bat

# 3. 문제 해결 후 재시도
```

---

## 📊 버전 히스토리 요약

| 버전 | 날짜 | 주요 변경사항 | 백업 상태 |
|---|---|---|---|
| v7.3 | 2026-02-06 | 재현성 보장 (Deterministic) | ✅ |
| v7.2 | 2026-02-05 | 적중률 우선 최적화 | ✅ |
| v7.1 | 2026-02-05 | 10% 연간 목표 | ❌ |
| v7.0 | 2026-02-05 | 기초 코드 복원 | ❌ |

**참고**: v7.0, v7.1은 백업 원칙 수립 전이라 백업 없음

---

## 🎯 다음 버전 계획

### v7.4 (예정)
- [ ] 적중률 60% 달성 검증
- [ ] 재현성 테스트 완료
- [ ] 성능 최적화

### v8.0 (예정)
- [ ] 새로운 기능 추가 시
- [ ] 아키텍처 변경 시
