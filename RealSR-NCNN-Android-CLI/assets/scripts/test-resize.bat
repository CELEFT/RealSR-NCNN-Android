@echo off

:: Test script for resize-ncnn.exe
:: This script calls test-all.bat to test resize-ncnn.exe

:: Call the main test script with resize-ncnn.exe as parameter
call "%~dp0test-all.bat" resize-ncnn.exe

pause
