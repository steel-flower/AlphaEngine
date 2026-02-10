# 이메일 알림 설정 가이드 (1분 완성!)

## ✅ 가장 간단한 방법!

---

## 📧 Gmail 사용 (추천)

### 준비물
```
✅ Gmail 계정 (기존 계정 사용 가능)
✅ 1분
```

---

## 1단계: Gmail 앱 비밀번호 발급 (1분)

### **방법 A: 2단계 인증이 이미 켜져 있는 경우**

```
1. https://myaccount.google.com/apppasswords 접속
2. Google 계정 로그인
3. 앱 선택: "메일"
4. 기기 선택: "Windows 컴퓨터"
5. [생성] 클릭
6. 16자리 비밀번호 표시됨
   예: abcd efgh ijkl mnop
7. 복사 (공백 포함 또는 제거, 둘 다 가능)
```

### **방법 B: 2단계 인증이 꺼져 있는 경우**

#### **Step 1: 2단계 인증 켜기**
```
1. https://myaccount.google.com/security 접속
2. "2단계 인증" 찾기
3. [시작하기] 클릭
4. 휴대전화 번호 입력
5. SMS 인증
6. 완료
```

#### **Step 2: 앱 비밀번호 발급**
```
1. https://myaccount.google.com/apppasswords 접속
2. 위의 "방법 A" 진행
```

---

## 2단계: 설정 파일 작성 (30초)

### 파일 위치
```
AlphaEngine 폴더 → email_config.json
```

### 내용 입력
```json
{
    "sender_email": "your_email@gmail.com",
    "sender_password": "abcdefghijklmnop",
    "receiver_email": "your_email@gmail.com",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}
```

### 설명
```
sender_email: 본인 Gmail 주소
sender_password: 1단계에서 발급받은 16자리 비밀번호
receiver_email: 알림 받을 이메일 (보통 sender와 동일)
smtp_server: smtp.gmail.com (변경 안 함)
smtp_port: 587 (변경 안 함)
```

### 예시
```json
{
    "sender_email": "hong@gmail.com",
    "sender_password": "abcdefghijklmnop",
    "receiver_email": "hong@gmail.com",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}
```

---

## 3단계: 테스트 (10초)

### 실행
```
AlphaEngine 폴더 → run_email_test.bat 더블클릭
```

### 성공 시
```
Gmail 받은편지함 확인
→ "✅ Alpha Engine 시스템 테스트" 이메일 도착
```

---

## ✅ 완료!

**총 소요 시간: 1분 30초**

이제 `run_monitor_v7.7.bat`를 실행하면 이메일로 알림을 받습니다!

---

## 📧 알림 예시

### 제목
```
🔔 Alpha Engine 매수 신호 - KODEX 코스피
```

### 내용
```
Alpha Engine 매수 신호

종목: KODEX 코스피 (226490)
현재가: 52,380원
진입가: 52,400원
목표가: 55,200원 (+5.3%)
손절가: 51,100원 (-2.5%)

AI 점수: 0.58
기술 점수: 0.42

시간: 2026-02-07 09:05:23

---
Alpha Engine v7.7
```

---

## 🆘 문제 해결

### "앱 비밀번호 메뉴가 안 보여요"
```
→ 2단계 인증을 먼저 켜야 합니다
→ https://myaccount.google.com/security
→ "2단계 인증" 시작하기
```

### "이메일이 안 와요"
```
1. email_config.json 확인
   - sender_email 정확한지
   - sender_password 16자리 맞는지
   
2. Gmail 스팸함 확인

3. 앱 비밀번호 재발급
```

### "로그인 실패"
```
→ 일반 Gmail 비밀번호가 아닌
→ "앱 비밀번호" 16자리를 사용해야 합니다!
```

---

## 💡 다른 이메일 사용

### Naver 메일
```json
{
    "sender_email": "your_id@naver.com",
    "sender_password": "naver_password",
    "receiver_email": "your_id@naver.com",
    "smtp_server": "smtp.naver.com",
    "smtp_port": 587
}
```

### Daum 메일
```json
{
    "sender_email": "your_id@daum.net",
    "sender_password": "daum_password",
    "receiver_email": "your_id@daum.net",
    "smtp_server": "smtp.daum.net",
    "smtp_port": 465
}
```

---

## 🎉 이메일의 장점

✅ 가입 불필요 (기존 이메일 사용)
✅ 설정 1분
✅ 복잡한 인증 없음
✅ 무료
✅ 안정적
✅ 모든 기기에서 확인 가능

**Happy Trading! 📈**
