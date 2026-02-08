# 🚀 Alpha Engine v7.7 - 빠른 시작 가이드

## ✅ 완료된 작업

Alpha Engine을 Streamlit Cloud에 배포할 수 있도록 다음 파일들이 생성되었습니다:

### 📁 생성된 파일

1. **`app.py`** - Streamlit 웹 애플리케이션
   - 로그인/인증 시스템
   - 실시간 대시보드
   - 종목 분석 및 백테스팅 결과
   - 설정 관리

2. **`requirements.txt`** - Python 패키지 의존성
   - Streamlit, Plotly 등 필요한 모든 패키지

3. **`.streamlit/config.toml`** - Streamlit 설정
   - 테마 및 서버 설정

4. **`.streamlit/secrets.toml.example`** - 비밀번호 예시
   - Streamlit Cloud Secrets 설정 참고용

5. **`.gitignore`** - Git 제외 파일
   - 민감 정보 보호

6. **`README_DEPLOY.md`** - 상세 배포 가이드
   - 단계별 배포 방법 설명

---

## 🖥️ 로컬에서 테스트

현재 Streamlit 앱이 실행 중입니다!

### 접속 방법

1. **웹 브라우저 열기**
2. **주소창에 입력**: `http://localhost:8501`
3. **비밀번호 입력**: `alpha2026` (기본값)

### 주요 기능

- **📊 대시보드**: 실시간 매매 신호 및 주요 지표
- **🔍 종목 분석**: 상세 기술 지표 및 AI 분석
- **📈 백테스팅 결과**: 성과 차트 및 통계
- **⚙️ 설정**: 종목 관리 및 알림 설정

---

## 🌐 Streamlit Cloud 배포 (3단계)

### Step 1: GitHub 저장소 생성

1. [github.com](https://github.com) 로그인
2. 우측 상단 `+` → `New repository`
3. 설정:
   - **Repository name**: `alpha-engine`
   - **Visibility**: **Private** (극소수 지인만 접근)
4. `Create repository` 클릭

### Step 2: 코드 업로드

#### 방법 A: GitHub Desktop (추천)

1. [GitHub Desktop](https://desktop.github.com) 다운로드 및 설치
2. `File` → `Add local repository`
3. Alpha Engine 폴더 선택: `C:\Users\user\Desktop\Antigravity\AlphaEngine`
4. `Create a repository` 클릭
5. 커밋 메시지 입력: `Initial commit - Alpha Engine v7.7`
6. `Commit to main` → `Publish repository` 클릭

#### 방법 B: Git 명령어

```powershell
cd C:\Users\user\Desktop\Antigravity\AlphaEngine
git init
git add .
git commit -m "Initial commit - Alpha Engine v7.7"
git remote add origin https://github.com/YOUR_USERNAME/alpha-engine.git
git branch -M main
git push -u origin main
```

### Step 3: Streamlit Cloud 배포

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. GitHub 계정으로 로그인
3. `New app` 클릭
4. 설정:
   - **Repository**: `YOUR_USERNAME/alpha-engine`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. `Deploy!` 클릭
6. 배포 완료 대기 (약 2-5분)

### Step 4: 비밀번호 설정

1. 배포된 앱 페이지에서 우측 상단 `⋮` → `Settings`
2. 좌측 메뉴 `Secrets` 선택
3. 다음 내용 입력:

```toml
password = "your_secure_password_here"
```

4. `Save` 클릭

---

## 🔐 보안 설정

### 비밀번호 변경

- Streamlit Cloud Secrets에서 `password` 값 변경
- 강력한 비밀번호 사용 권장

### 접근 제어

- **Private Repository**: GitHub 저장소를 Private으로 설정
- **비밀번호 공유**: 지인에게만 비밀번호 전달
- **URL 비공개**: 웹 앱 URL을 공개하지 않음

---

## 📱 사용 방법

### 웹 접속

1. **배포된 URL 접속** (예: `https://your-app.streamlit.app`)
2. **비밀번호 입력**
3. **종목 선택** (사이드바)
4. **대시보드 확인**

### 모바일 접속

- 스마트폰 브라우저에서도 동일한 URL로 접속 가능
- 반응형 디자인 지원

---

## 🔄 업데이트 방법

### 코드 수정 후 배포

1. **로컬에서 파일 수정** (예: `app.py`)
2. **GitHub에 푸시**:
   - GitHub Desktop: `Commit` → `Push origin`
   - Git: `git add . && git commit -m "Update" && git push`
3. **자동 재배포** (Streamlit Cloud가 자동 감지)

---

## ⚠️ 주의사항

### 무료 플랜 제한

- CPU/메모리 제한 있음
- 동시 사용자 제한적 (소수 사용자에게 적합)
- 장시간 실행 시 자동 종료 가능

### 최적화

- 캐싱 활용 (`@st.cache_data`)
- AI 학습 간소화 (epochs=10)
- 필요한 종목만 분석

---

## 🆘 문제 해결

### 배포 실패

- Streamlit Cloud 로그 확인
- `requirements.txt` 패키지 버전 확인

### 느린 실행

- 캐시 TTL 증가
- AI epochs 감소
- 분석 종목 수 제한

### 비밀번호 오류

- Streamlit Cloud Secrets 확인
- 대소문자 구분 확인

---

## 📞 지원

- **Streamlit 문서**: [docs.streamlit.io](https://docs.streamlit.io)
- **커뮤니티**: [discuss.streamlit.io](https://discuss.streamlit.io)

---

## 🎉 완료!

이제 Alpha Engine을 웹에서 사용할 수 있습니다!

**다음 단계:**
1. ✅ 로컬에서 테스트 (`http://localhost:8501`)
2. 📤 GitHub에 업로드
3. 🚀 Streamlit Cloud 배포
4. 🔐 비밀번호 설정
5. 🎯 지인과 공유

**배포 성공을 기원합니다!** 🚀
