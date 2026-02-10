"""
Alpha Engine v7.7 - 이메일 알림 모듈
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from datetime import datetime


class EmailNotifier:
    def __init__(self, config_file="email_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
    def load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 기본 설정 파일 생성
            default_config = {
                "sender_email": "YOUR_EMAIL@gmail.com",
                "sender_password": "YOUR_APP_PASSWORD",
                "receiver_email": "YOUR_EMAIL@gmail.com",
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print(f"[알림] {self.config_file} 파일이 생성되었습니다.")
            print("Gmail 주소와 앱 비밀번호를 입력해주세요.")
            return default_config
    
    def send_email(self, subject, body):
        """이메일 전송"""
        sender = self.config.get('sender_email')
        password = self.config.get('sender_password')
        receiver = self.config.get('receiver_email')
        
        if not sender or not password or 'YOUR_' in sender:
            print("[오류] 이메일 설정이 완료되지 않았습니다.")
            print(f"email_config.json 파일을 확인해주세요.")
            return False
        
        try:
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = receiver
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Gmail SMTP 서버 연결
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(sender, password)
            
            # 이메일 전송
            server.send_message(msg)
            server.quit()
            
            print(f"[성공] 이메일 전송 완료: {subject}")
            return True
            
        except Exception as e:
            print(f"[오류] 이메일 전송 실패: {e}")
            return False
    
    def send_buy_signal(self, ticker, name, current_price, entry_price, target_price, stop_loss, ai_score, tech_score):
        """매수 신호 알림"""
        profit_pct = (target_price / entry_price - 1) * 100
        loss_pct = (1 - stop_loss / entry_price) * 100
        
        subject = f"🔔 Alpha Engine 매수 신호 - {name}"
        
        body = f"""Alpha Engine 매수 신호

종목: {name} ({ticker})
현재가: {current_price:,.0f}원
진입가: {entry_price:,.0f}원
목표가: {target_price:,.0f}원 (+{profit_pct:.1f}%)
손절가: {stop_loss:,.0f}원 (-{loss_pct:.1f}%)

AI 점수: {ai_score:.2f}
기술 점수: {tech_score:.2f}

시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Alpha Engine v7.7
"""
        
        return self.send_email(subject, body)
    
    def send_sell_signal(self, ticker, name, entry_price, current_price, reason):
        """청산 신호 알림"""
        profit_pct = (current_price / entry_price - 1) * 100
        
        subject = f"🔴 Alpha Engine 청산 신호 - {name} ({profit_pct:+.2f}%)"
        
        body = f"""Alpha Engine 청산 신호

종목: {name} ({ticker})
진입가: {entry_price:,.0f}원
현재가: {current_price:,.0f}원
수익률: {profit_pct:+.2f}%

사유: {reason}

시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Alpha Engine v7.7
"""
        
        return self.send_email(subject, body)
    
    def send_daily_summary(self, summary_text):
        """일일 요약 알림"""
        subject = "📊 Alpha Engine 일일 요약"
        
        body = f"""Alpha Engine 일일 요약

{summary_text}

시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Alpha Engine v7.7
"""
        
        return self.send_email(subject, body)


# 테스트 코드
if __name__ == "__main__":
    notifier = EmailNotifier()
    
    print("\n" + "="*50)
    print("이메일 알림 테스트")
    print("="*50)
    
    # 테스트 메시지 전송
    subject = "[OK] Alpha Engine 시스템 테스트"
    body = """Alpha Engine v7.7 이메일 알림 시스템이 정상 작동 중입니다!

설정이 완료되었습니다!

---
Alpha Engine v7.7
"""
    
    notifier.send_email(subject, body)
