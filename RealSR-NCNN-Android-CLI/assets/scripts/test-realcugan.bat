@echo off

:: Test script for realcugan-ncnn.exe
:: This script calls test-all.bat to test realcugan-ncnn.exe

:: Call the main test script with realcugan-ncnn.exe as parameter
call "%~dp0test-all.bat" realcugan-ncnn.exe

pause
