@echo off
title AI Text Generator - LLM Fundamentals
echo ========================================================
echo Starting AI Text Generator using DistilGPT-2 (Gradio UI)
echo ========================================================
echo.

set PYTHON_BIN="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if exist %PYTHON_BIN% (
    %PYTHON_BIN% app.py
) else (
    python app.py
)

pause
