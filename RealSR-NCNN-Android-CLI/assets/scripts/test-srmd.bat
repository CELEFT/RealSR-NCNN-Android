@echo off

:: Test script for srmd-ncnn.exe
:: This script calls test-all.bat to test srmd-ncnn.exe

:: Call the main test script with srmd-ncnn.exe as parameter
call "%~dp0test-all.bat" srmd-ncnn.exe

pause
