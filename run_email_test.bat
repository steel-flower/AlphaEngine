@echo off
chcp 65001 > nul
title Alpha Engine v7.7 - 이메일 알림 테스트

echo.
echo ======================================================
echo    Alpha Engine v7.7 - 이메일 알림 테스트
echo ======================================================
echo.

python email_notifier.py

echo.
echo Gmail 받은편지함에서 메시지를 확인하세요!
echo.
pause
