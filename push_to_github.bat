@echo off
title Push Employee Attrition ML to GitHub
color 0B
echo ================================================================
echo   CONNECTING AND PUSHING CODE TO GITHUB REPOSITORY
echo   Repo: https://github.com/freefirepunu44-pixel/Employee-Attrition-ML
echo ================================================================
echo.
cd /d "%~dp0"
echo Running git push -u origin main --force...
echo.
echo * Note: A browser window or login prompt will open. 
echo   Click "Sign in with your browser" to authenticate.
echo.
git push -u origin main --force
echo.
if %ERRORLEVEL% EQU 0 (
    color 0A
    echo ================================================================
    echo   SUCCESS! All files and commits are now on GitHub!
    echo ================================================================
) else (
    color 0C
    echo ================================================================
    echo   Push failed or was cancelled. Please check the error above.
    echo ================================================================
)
echo.
pause
