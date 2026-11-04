@echo off
cd /d "%~dp0"
set "POOL=stratum+tcp://pearl-sg1.luckypool.io:3360"
set "WALLET=YOUR_PEARL_WALLET"
set "WORKER=5090D-experiment"
set "MODE=int8-tall"
if "%WALLET%"=="YOUR_PEARL_WALLET" (
 echo Please set your Pearl wallet in this file before starting.
 pause
 exit /b 1
)
cppminer.exe --backend cuda --pool "%POOL%" --wallet "%WALLET%" --worker "%WORKER%" --tensor-mode "%MODE%" --binary-a --m 128 --n 16 --verify --no-fee
pause
