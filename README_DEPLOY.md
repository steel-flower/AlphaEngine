# 📈 Alpha Engine v7.7 - Streamlit Cloud 배포 가이드

## 🎯 개요

Alpha Engine을 Streamlit Cloud에 무료로 배포하여 웹에서 접근 가능하게 만드는 가이드입니다.

---

## 📋 배포 전 준비사항

### 1. 필요한 파일 확인

다음 파일들이 프로젝트 폴더에 있는지 확인하세요:

- ✅ `app.py` - Streamlit 웹 앱
- ✅ `alpha_engine_sigma.py` - Alpha Engine 코어
- ✅ `requirements.txt` - 패키지 의존성
- ✅ `assets.json` - 종목 리스트
- ✅ `.streamlit/config.toml` - Streamlit 설정
- ✅ `.gitignore` - Git 제외 파일

### 2. GitHub 계정 준비

- GitHub 계정이 없다면 [github.com](https://github.com)에서 무료 가입
- 기존 계정이 있다면 로그인

---

## 🚀 배포 단계

### Step 1: GitHub 저장소 생성

1. **GitHub에 로그인** 후 우측 상단 `+` 버튼 클릭
2. **"New repository"** 선택
3. 저장소 설정:
   - **Repository name**: `alpha-engine` (원하는 이름)
   - **Description**: `AI-based Trading Signal System`
   - **Visibility**: **Private** (극소수 지인만 접근하려면 Private 선택)
   - **Initialize this repository with**: 체크 안 함
4. **"Create repository"** 클릭

### Step 2: 로컬 프로젝트를 GitHub에 업로드

#### 방법 A: GitHub Desktop 사용 (초보자 추천)

1. **GitHub Desktop 다운로드 및 설치**
   - [desktop.github.com](https://desktop.github.com)에서 다운로드

2. **GitHub Desktop에서 저장소 추가**
   - `File` → `Add local repository`
   - Alpha Engine 폴더 선택 (`C:\Users\user\Desktop\Antigravity\AlphaEngine`)
   - "Create a repository" 클릭

3. **파일 커밋 및 푸시**
   - 좌측 하단에 커밋 메시지 입력: `Initial commit - Alpha Engine v7.7`
   - `Commit to main` 클릭
   - 상단 `Publish repository` 클릭
   - Private 체크 후 `Publish repository` 클릭

#### 방법 B: Git 명령어 사용 (고급 사용자)

PowerShell에서 다음 명령어 실행:

```powershell
cd C:\Users\user\Desktop\Antigravity\AlphaEngine

# Git 초기화
git init

# 파일 추가
git add .

# 커밋
git commit -m "Initial commit - Alpha Engine v7.7"

# GitHub 저장소 연결 (YOUR_USERNAME을 본인 GitHub 아이디로 변경)
git remote add origin https://github.com/YOUR_USERNAME/alpha-engine.git

# 푸시
git branch -M main
git push -u origin main
```

### Step 3: Streamlit Cloud 배포

1. **Streamlit Cloud 접속**
   - [share.streamlit.io](https://share.streamlit.io) 방문
   - GitHub 계정으로 로그인

2. **새 앱 배포**
   - `New app` 버튼 클릭
   - 설정:
     - **Repository**: `YOUR_USERNAME/alpha-engine` 선택
     - **Branch**: `main`
     - **Main file path**: `app.py`
   - `Deploy!` 클릭

3. **배포 대기**
   - 약 2-5분 소요
   - 배포 로그에서 진행 상황 확인

### Step 4: 비밀번호 설정 (Secrets)

1. **Streamlit Cloud 앱 설정**
   - 배포된 앱 페이지에서 우측 상단 `⋮` (메뉴) 클릭
   - `Settings` 선택

2. **Secrets 추가**
   - 좌측 메뉴에서 `Secrets` 선택
   - 다음 내용 입력:

```toml
password = "your_secure_password_here"
```

   - `your_secure_password_here`를 원하는 비밀번호로 변경
   - `Save` 클릭

3. **앱 재시작**
   - 자동으로 재시작되며, 새 비밀번호가 적용됨

---

## 🔐 접근 제어 (극소수 지인만 접근)

### 방법 1: 비밀번호 공유 (기본)

- Streamlit Secrets에 설정한 비밀번호를 지인에게만 공유
- 웹 앱 URL을 알아도 비밀번호 없이는 접근 불가

### 방법 2: GitHub Private Repository (추가 보안)

- GitHub 저장소를 Private으로 설정 (Step 1에서 설정)
- Streamlit Cloud는 Private 저장소도 배포 가능
- 검색 엔진에 노출되지 않음

### 방법 3: IP 화이트리스트 (고급)

- Streamlit Cloud 유료 플랜에서 제공
- 특정 IP 주소만 접근 허용

---

## 📱 사용 방법

### 웹 앱 접속

1. **URL 확인**
   - Streamlit Cloud에서 배포 완료 후 URL 제공
   - 예: `https://your-app-name.streamlit.app`

2. **로그인**
   - 웹 브라우저에서 URL 접속
   - Secrets에 설정한 비밀번호 입력

3. **종목 분석**
   - 사이드바에서 종목 선택
   - 대시보드에서 실시간 신호 확인

### 모바일 접속

- 스마트폰 브라우저에서도 동일한 URL로 접속 가능
- 반응형 디자인으로 모바일에서도 사용 가능

---

## 🔄 업데이트 방법

### 코드 수정 후 배포

1. **로컬에서 코드 수정**
   - `app.py` 또는 `alpha_engine_sigma.py` 수정

2. **GitHub에 푸시**
   - GitHub Desktop: `Commit` → `Push origin`
   - Git 명령어:
     ```powershell
     git add .
     git commit -m "Update: 변경 내용 설명"
     git push
     ```

3. **자동 재배포**
   - Streamlit Cloud가 자동으로 감지하여 재배포
   - 약 1-2분 소요

---

## ⚠️ 주의사항

### 무료 플랜 제한

- **리소스**: CPU/메모리 제한 있음
- **동시 사용자**: 제한적 (소수 사용자에게 적합)
- **실행 시간**: 장시간 실행 시 자동 종료 가능

### 최적화 팁

1. **캐싱 활용**
   - `@st.cache_data` 데코레이터로 분석 결과 캐싱
   - 1시간 TTL 설정으로 불필요한 재계산 방지

2. **AI 학습 간소화**
   - 웹 앱에서는 간단한 학습만 수행 (epochs=10)
   - 전체 최적화는 로컬에서 실행 후 결과만 표시

3. **데이터 로딩 최소화**
   - 필요한 종목만 분석
   - 캐시 활용으로 중복 로딩 방지

### 보안 주의사항

- **비밀번호 관리**: 강력한 비밀번호 사용
- **민감 정보**: `email_config.json` 등은 GitHub에 업로드하지 않음 (`.gitignore`에 포함됨)
- **API 키**: Streamlit Secrets에만 저장, 코드에 하드코딩 금지

---

## 🆘 문제 해결

### 배포 실패 시

1. **로그 확인**
   - Streamlit Cloud 배포 페이지에서 로그 확인
   - 에러 메시지 확인

2. **일반적인 문제**
   - `requirements.txt` 패키지 버전 충돌 → 버전 명시 제거 또는 수정
   - 파일 경로 오류 → 상대 경로 사용 확인
   - 메모리 부족 → AI 모델 간소화 또는 데이터 크기 축소

### 느린 실행 속도

- **원인**: 무료 플랜의 리소스 제한
- **해결**:
  - 캐싱 강화 (`@st.cache_data` TTL 증가)
  - AI 학습 epochs 감소
  - 분석 종목 수 제한

### 비밀번호 변경

1. Streamlit Cloud 앱 설정 → Secrets
2. `password` 값 변경
3. 저장 후 자동 재시작

---

## 📞 지원

문제가 발생하면:

1. **Streamlit 문서**: [docs.streamlit.io](https://docs.streamlit.io)
2. **커뮤니티**: [discuss.streamlit.io](https://discuss.streamlit.io)
3. **GitHub Issues**: 저장소에서 이슈 생성

---

## 🎉 완료!

이제 Alpha Engine을 웹에서 사용할 수 있습니다!

**배포된 URL을 지인에게 공유하고, 비밀번호를 안전하게 전달하세요.** 🚀
