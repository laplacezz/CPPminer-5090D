@echo off
cd /d "%~dp0"
set "POOL=YOUR_POOL:PORT"
set "WALLET=YOUR_PEARL_WALLET.worker"
if "%POOL%"=="YOUR_POOL:PORT" (
 echo Please edit POOL and WALLET in this file before starting.
 pause
 exit /b 1
)
cppminer.exe --backend cuda --pool "%POOL%" --wallet "%WALLET%" --tensor-mode int8
pause
