@echo off

:: Test script for waifu2x-ncnn.exe
:: This script calls test-all.bat to test waifu2x-ncnn.exe

:: Call the main test script with waifu2x-ncnn.exe as parameter
call "%~dp0test-all.bat" waifu2x-ncnn.exe

pause
