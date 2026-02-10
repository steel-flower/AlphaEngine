"""
Alpha Engine Sigma v3.2 - Real-time Email Alert Module
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
        """Load or Create Configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            default_config = {
                "sender_email": "frederic.jeon@gmail.com",
                "sender_password": "YOUR_APP_PASSWORD",
                "receiver_email": "frederic.jeon@gmail.com",
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            return default_config
    
    def send_email(self, subject, body):
        """Send Email via SMTP (Supports multiple recipients)"""
        sender = self.config.get('sender_email')
        password = self.config.get('sender_password')
        receiver_input = self.config.get('receiver_email', '')
        
        if not sender or not password or 'YOUR_' in password:
            print("[Error] Email configuration not complete. Please check email_config.json")
            return False
        
        # Split multiple recipients by comma if any
        recipients = [r.strip() for r in receiver_input.split(',')] if ',' in receiver_input else [receiver_input.strip()]
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
            server.quit()
            
            print(f" [Email] Successfully sent to {len(recipients)} recipients: {subject}")
            return True
        except Exception as e:
            print(f" [Email] Failed: {e}")
            return False
    
    def send_buy_signal(self, ticker, name, current_price, entry_price, target_price, stop_loss, ai_score, tech_score):
        """Buy Signal Alert"""
        profit_pct = (target_price / entry_price - 1) * 100
        loss_pct = (1 - stop_loss / entry_price) * 100
        
        subject = f"🔔 [Alpha Engine v3.2] 매수 권장 - {name}"
        
        body = f"""Alpha Engine v3.2 Master Precision 매수 신호 알림

[분석 정보]
- 종목명: {name} ({ticker})
- 현재가: {current_price:,.0f}원
- 진입 권장가: {entry_price:,.0f}원 이하

[전략 가이드]
- 목표가(익절): {target_price:,.0f}원 (+{profit_pct:.1f}%)
- 손절가(방어): {stop_loss:,.0f}원 (-{loss_pct:.1f}%)
- AI Score: {ai_score:.2f} (5% 기대이익 가드 통과)

[발생 시간]
- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Alpha Engine Sigma v3.2 Master Precision
This message is automated by the monitoring system.
"""
        return self.send_email(subject, body)
    
    def send_sell_signal(self, ticker, name, entry_price, current_price, reason):
        """Exit/Wait Signal Alert"""
        profit_pct = (current_price / entry_price - 1) * 100
        
        subject = f"🔴 [Alpha Engine v3.2] 관망 전환/청산 - {name}"
        
        body = f"""Alpha Engine v3.2 Master Precision 포지션 종료/관망 알림

[분석 정보]
- 종목명: {name} ({ticker})
- 현재가: {current_price:,.0f}원
- 수익률(진입가 대비): {profit_pct:+.2f}%

[사유]
- {reason}

[발생 시간]
- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Alpha Engine Sigma v3.2 Master Precision
This message is automated by the monitoring system.
"""
        return self.send_email(subject, body)
    
    def send_daily_summary(self, summary_text):
        """일일 요약 알림"""
        subject = "📊 Alpha Engine v3.2 일일 요약"
        
        body = f"""Alpha Engine v3.2 Master Precision 일일 요약

{summary_text}

시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Alpha Engine Sigma v3.2 Master Precision
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
