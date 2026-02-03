@echo off

:: Test script for realsr-ncnn.exe
:: This script calls test-all.bat to test realsr-ncnn.exe

:: Call the main test script with realsr-ncnn.exe as parameter
call "%~dp0test-all.bat" realsr-ncnn.exe

pause
