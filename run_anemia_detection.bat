@echo off
title Anemia Detection System
color 0A

echo.
echo ╔══════════════════════════════════════════╗
echo ║       ANEMIA DETECTION SYSTEM           ║
echo ║   Using DenseNet + VGG16 + InceptionV3  ║
echo ╚══════════════════════════════════════════╝
echo.

:: Activate virtual environment
call anemia_env\Scripts\activate.bat

:ask_image
echo.
echo  Enter the FULL PATH of your eye image below.
echo  Example: C:\Users\princ\Desktop\my_eye.jpg
echo.
set /p IMAGE_PATH= 

:: Check if user entered anything
if "%IMAGE_PATH%"=="" (
    echo.
    echo  ERROR: You did not enter any path! Please try again.
    goto ask_image
)

:: Remove surrounding quotes if user added them
set IMAGE_PATH=%IMAGE_PATH:"=%

:: Check if file exists
if not exist "%IMAGE_PATH%" (
    echo.
    echo  ERROR: File not found at: %IMAGE_PATH%
    echo  Please check the path and try again.
    goto ask_image
)

echo.
echo  Analyzing image... Please wait...
echo.

:: Run prediction
python predict.py --image "%IMAGE_PATH%"

echo.
echo ══════════════════════════════════════════
echo.

:ask_again
set /p AGAIN= Do you want to test another image? (yes/no): 
if /i "%AGAIN%"=="yes" goto ask_image
if /i "%AGAIN%"=="y"   goto ask_image
if /i "%AGAIN%"=="no"  goto end
if /i "%AGAIN%"=="n"   goto end
echo Please type yes or no.
goto ask_again

:end
echo.
echo  Thank you for using Anemia Detection System!
echo  Always consult a doctor for medical advice.
echo.
pause
