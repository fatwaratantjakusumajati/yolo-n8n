@echo off
:: ======================================================================
:: SETUP NSSM — Auto-start CCTV Alert saat Windows startup
:: Jalankan sebagai Administrator!
:: ======================================================================

echo ============================================================
echo  SETUP NSSM — CCTV Alert System
echo  Jalankan sebagai Administrator!
echo ============================================================
echo.

:: Cek apakah dijalankan sebagai Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Harap jalankan sebagai Administrator!
    echo Klik kanan file ini, pilih "Run as administrator"
    pause
    exit /b 1
)

:: ======================================================================
:: KONFIGURASI
:: ======================================================================
@REM set NSSM_PATH=C:\Users\ASUS NUC 13 PRO\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe
@REM set PYTHON_PATH=C:\Users\ASUS NUC 13 PRO\AppData\Local\Programs\Python\Python314\python.exe
@REM set SCRIPT_PATH=C:\Users\ASUS NUC 13 PRO\Documents\Fatwa\code\n8n-yolo\cctv-monitor.py
@REM set WORK_DIR=C:\Users\ASUS NUC 13 PRO\Documents\Fatwa\code\n8n-yolo
@REM set SERVICE_CCTV=CCTVAlertSystem

set "NSSM_PATH=C:\Users\ASUS NUC 13 PRO\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe"
set "PYTHON_PATH=C:\Users\ASUS NUC 13 PRO\AppData\Local\Programs\Python\Python314\python.exe"
set "SCRIPT_PATH=C:\Users\ASUS NUC 13 PRO\Documents\Fatwa\code\n8n-yolo\cctv-monitor.py"
set "WORK_DIR=C:\Users\ASUS NUC 13 PRO\Documents\Fatwa\code\n8n-yolo"
set SERVICE_CCTV=CCTVAlertSystem

:: ======================================================================
:: CEK NSSM
:: ======================================================================
if not exist "%NSSM_PATH%" (
    echo [ERROR] NSSM tidak ditemukan!
    pause
    exit /b 1
)
echo [OK] NSSM ditemukan

:: ======================================================================
:: HAPUS SERVICE LAMA (kalau ada)
:: ======================================================================
echo.
echo [INFO] Menghapus service lama (kalau ada)...
"%NSSM_PATH%" stop %SERVICE_CCTV% >nul 2>&1
"%NSSM_PATH%" remove %SERVICE_CCTV% confirm >nul 2>&1
echo [OK] Service lama dihapus

:: ======================================================================
:: BUAT FOLDER LOGS
:: ======================================================================
if not exist "%WORK_DIR%\logs" mkdir "%WORK_DIR%\logs"
echo [OK] Folder logs dibuat

:: ======================================================================
:: INSTALL SERVICE: CCTV ALERT SYSTEM
:: ======================================================================
echo.
echo [INFO] Membuat service %SERVICE_CCTV%...

"%NSSM_PATH%" install %SERVICE_CCTV% "%PYTHON_PATH%"
"%NSSM_PATH%" set %SERVICE_CCTV% AppParameters "%SCRIPT_PATH%"
"%NSSM_PATH%" set %SERVICE_CCTV% AppDirectory "%WORK_DIR%"
"%NSSM_PATH%" set %SERVICE_CCTV% DisplayName "CCTV Alert System"
"%NSSM_PATH%" set %SERVICE_CCTV% Description "YOLO Detection + Monitor 22 Kamera"
"%NSSM_PATH%" set %SERVICE_CCTV% Start SERVICE_AUTO_START

:: Restart otomatis kalau crash
"%NSSM_PATH%" set %SERVICE_CCTV% AppRestartDelay 10000
"%NSSM_PATH%" set %SERVICE_CCTV% AppThrottle 5000
"%NSSM_PATH%" set %SERVICE_CCTV% AppExit Default Restart

:: Log output
"%NSSM_PATH%" set %SERVICE_CCTV% AppStdout "%WORK_DIR%\logs\cctv_stdout.log"
"%NSSM_PATH%" set %SERVICE_CCTV% AppStderr "%WORK_DIR%\logs\cctv_stderr.log"
"%NSSM_PATH%" set %SERVICE_CCTV% AppRotateFiles 1
"%NSSM_PATH%" set %SERVICE_CCTV% AppRotateBytes 10485760

echo [OK] Service %SERVICE_CCTV% berhasil dibuat!

:: ======================================================================
:: START SERVICE
:: ======================================================================
echo.
echo [INFO] Menjalankan service...
"%NSSM_PATH%" start %SERVICE_CCTV%

:: ======================================================================
:: CEK STATUS
:: ======================================================================
echo.
echo [INFO] Status service:
"%NSSM_PATH%" status %SERVICE_CCTV%

echo.
echo ============================================================
echo  SETUP SELESAI!
echo ============================================================
echo.
echo  Service yang dibuat:
echo  - %SERVICE_CCTV% — CCTV Alert + Monitor 22 kamera
echo.
echo  Otomatis jalan saat Windows startup.
echo  Kalau crash, otomatis restart dalam 10 detik.
echo.
echo  Untuk cek status: services.msc
echo  Untuk stop: jalankan stop_all.bat
echo.
pause