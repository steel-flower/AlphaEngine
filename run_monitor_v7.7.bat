@echo off
setlocal enabledelayedexpansion
title Alpha Engine v7.7 Hybrid Monitor & Deployer

:: =============================================================================
:: Alpha Engine v7.7 하이브리드 자동화 시스템 (v2.0)
:: 1. 10분 간격 정밀 분석 및 이메일 알림 발송
:: 2. 웹 앱용 정밀 데이터 갱신
:: 3. GitHub 자동 Push를 통한 웹 실시간 배포
:: =============================================================================

echo.
echo  [SYSTEM] Alpha Engine v7.7 Hybrid System 시작 중...
echo  [SYSTEM] 이메일 알림 및 웹 배포가 통합되었습니다.
echo.

:loop
echo  [%DATE% %TIME%] >>> 정밀 분석 및 모니터링 시작...

:: 1단계: 로컬 분석 및 이메일 발송 & 웹 데이터 생성
:: (run_quick_update()를 사용하여 저장된 최적값을 호출하고 빠르게 갱신합니다)
python -c "from alpha_engine_sigma import AlphaEngineSigma; import json; f=open('assets.json','r',encoding='utf-8'); assets=json.load(f); f.close(); [AlphaEngineSigma(a['ticker'], a['name']).run_quick_update() for a in assets]"

if %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] 분석 중 오류 발생. 다음 주기에 재시도합니다.
    goto wait
)

:: 2단계: 신호 모니터링 및 이메일 알림 (별도 모니터링 스크립트 실행)
:: signal_monitor_v7.7.py가 이메일을 담당함
python signal_monitor_v7.7.py

:: 3단계: GitHub 자동 배포 (웹 반영)
echo  [DEPLOY] 분석 결과를 웹 서버(GitHub)로 전송 중...
git add web_data_*.json ticker_blueprints.json assets.json
git commit -m "Auto-update: Alpha Engine v7.7 Analysis Results (%DATE% %TIME%)"
git push origin main

echo.
echo  [SUCCESS] 분석 완료, 이메일 체크 완료, 웹 업데이트 완료!
echo  [WAIT] 5분 대기 후 다음 주기를 시작합니다...
echo  [INFO] 중단하려면 Ctrl+C를 누르세요.

:wait
timeout /t 300 /nobreak > nul
goto loop
