@echo off
setlocal enabledelayedexpansion

:: Set base paths
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "WIN_X64_DIR=%ROOT_DIR%\Win-x64"
set "INPUT_DIR=%ROOT_DIR%\input"
set "OUTPUT_DIR=%ROOT_DIR%\output"

:: Change to Win-x64 directory
cd /d "%WIN_X64_DIR%" || (
echo Cannot change to %WIN_X64_DIR%
echo Please ensure the directory exists
goto :eof
)

:: Clean output directory
echo Cleaning output directory...
del /f /q "%OUTPUT_DIR%\*" >nul 2>&1

:: Check if input directory has files
if not exist "%INPUT_DIR%\*" (
echo No files in input directory %INPUT_DIR%
echo Please put test images in the input directory
goto :eof
)

:: Show help information
if "%1" == "help" goto :show_help
if "%1" == "--help" goto :show_help
if "%1" == "-h" goto :show_help

:: Check if program is specified
if "%1" neq "" (
    set "TARGET_PROGRAM=%1"
    goto :test_single_program
) else (
    goto :test_all_programs
)

:test_all_programs
echo Testing all programs...
echo.

call :test_resize
call :test_realcugan
call :test_realsr
call :test_srmd
call :test_waifu2x

echo.
echo All tests completed!
echo Results saved in %OUTPUT_DIR%
goto :eof

:test_single_program
echo Testing single program: %TARGET_PROGRAM%
echo.

if "%TARGET_PROGRAM%" == "resize-ncnn.exe" (
    call :test_resize
) else if "%TARGET_PROGRAM%" == "realcugan-ncnn.exe" (
    call :test_realcugan
) else if "%TARGET_PROGRAM%" == "realsr-ncnn.exe" (
    call :test_realsr
) else if "%TARGET_PROGRAM%" == "srmd-ncnn.exe" (
    call :test_srmd
) else if "%TARGET_PROGRAM%" == "waifu2x-ncnn.exe" (
    call :test_waifu2x
) else (
    echo Program not found: %TARGET_PROGRAM%
echo Please check the program name
)

echo.
echo Test completed!
echo Results saved in %OUTPUT_DIR%
goto :eof

:test_resize
if not exist "resize-ncnn.exe" (
    echo Program resize-ncnn.exe does not exist, skipping test
echo.
    goto :eof
)

echo Testing program: resize-ncnn.exe

:: Test parameter groups
set "INDEX=1"
call :run_test "resize-ncnn.exe" "-s 2 -m bicubic" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 0.5 -m avir" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 2 -m bilinear" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 2 -m nearest" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 2 -m avir-lancir" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 2 -m de-nearest" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 2 -m de-nearest2" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 2 -m de-nearest3" %INDEX%
set /a INDEX+=1
call :run_test "resize-ncnn.exe" "-s 0 -m perfectpixel" %INDEX%

echo.
goto :eof

:test_realcugan
if not exist "realcugan-ncnn.exe" (
    echo Program realcugan-ncnn.exe does not exist, skipping test
echo.
    goto :eof
)

echo Testing program: realcugan-ncnn.exe

:: Test parameter groups
set "INDEX=1"
call :run_test "realcugan-ncnn.exe" "-m models-nose -s 2 -n 0" %INDEX%
set /a INDEX+=1
call :run_test "realcugan-ncnn.exe" "-m models-se -s 2 -n -1" %INDEX%
set /a INDEX+=1
call :run_test "realcugan-ncnn.exe" "-m models-se -s 2 -n 0" %INDEX%
set /a INDEX+=1
call :run_test "realcugan-ncnn.exe" "-m models-pro -s 2 -n -1" %INDEX%
set /a INDEX+=1
call :run_test "realcugan-ncnn.exe" "-m models-pro -s 2 -n 0" %INDEX%

echo.
goto :eof

:test_realsr
if not exist "realsr-ncnn.exe" (
    echo Program realsr-ncnn.exe does not exist, skipping test
echo.
    goto :eof
)

echo Testing program: realsr-ncnn.exe

:: Test parameter groups
set "INDEX=1"
call :run_test "realsr-ncnn.exe" "-m models-Real-ESRGAN-anime" %INDEX%
set /a INDEX+=1
call :run_test "realsr-ncnn.exe" "-m models-Real-ESRGANv3-general -s 4" %INDEX%
set /a INDEX+=1
call :run_test "realsr-ncnn.exe" "-m models-Real-ESRGANv3-anime -s 3" %INDEX%

echo.
goto :eof

:test_srmd
if not exist "srmd-ncnn.exe" (
    echo Program srmd-ncnn.exe does not exist, skipping test
echo.
    goto :eof
)

echo Testing program: srmd-ncnn.exe

:: Test parameter groups
set "INDEX=1"
call :run_test "srmd-ncnn.exe" "-s 2" %INDEX%
set /a INDEX+=1
call :run_test "srmd-ncnn.exe" "-s 3" %INDEX%
set /a INDEX+=1
call :run_test "srmd-ncnn.exe" "-s 4" %INDEX%

echo.
goto :eof

:test_waifu2x
if not exist "waifu2x-ncnn.exe" (
    echo Program waifu2x-ncnn.exe does not exist, skipping test
echo.
    goto :eof
)

echo Testing program: waifu2x-ncnn.exe

:: Test parameter groups
set "INDEX=1"
call :run_test "waifu2x-ncnn.exe" "-s 2 -n 0" %INDEX%
set /a INDEX+=1
call :run_test "waifu2x-ncnn.exe" "-s 2 -n 1" %INDEX%
set /a INDEX+=1
call :run_test "waifu2x-ncnn.exe" "-s 2 -n 2" %INDEX%
set /a INDEX+=1
call :run_test "waifu2x-ncnn.exe" "-s 4 -n 0" %INDEX%
set /a INDEX+=1
call :run_test "waifu2x-ncnn.exe" "-s 4 -n 1" %INDEX%
set /a INDEX+=1
call :run_test "waifu2x-ncnn.exe" "-s 4 -n 2" %INDEX%

echo.
goto :eof

:run_test
set "PROGRAM=%~1"
set "PARAMS=%~2"
set "INDEX=%~3"

echo Test parameter group %INDEX%: %PARAMS%

for %%i in ("%INPUT_DIR%\*") do (
    set "INPUT_FILE=%%i"
    set "INPUT_NAME=%%~ni"
    set "INPUT_EXT=%%~xi"
    
    :: Generate output filename with parameters
    :: Process PARAMS to create a safe filename suffix
    set "PARAM_SUFFIX=%PARAMS%"
    set "PARAM_SUFFIX=!PARAM_SUFFIX:-s =s!"
    set "PARAM_SUFFIX=!PARAM_SUFFIX:-m =m!"
    set "PARAM_SUFFIX=!PARAM_SUFFIX:-n =n!"
    set "PARAM_SUFFIX=!PARAM_SUFFIX: =_!"
    set "PARAM_SUFFIX=!PARAM_SUFFIX:/=-!"
    
    :: Generate output filename with index
    set "OUTPUT_NAME=!PROGRAM:~0,-4!_%INDEX%_!PARAM_SUFFIX!_!INPUT_NAME!!INPUT_EXT!"
    set "OUTPUT_FILE=%OUTPUT_DIR%\!OUTPUT_NAME!"
    
    :: Run test
    echo Processing file: !INPUT_NAME!!INPUT_EXT!
    "%PROGRAM%" %PARAMS% -i "!INPUT_FILE!" -o "!OUTPUT_FILE!"
    
    if errorlevel 1 (
        echo Test failed: !PROGRAM! %PARAMS%
    ) else (
        echo Test succeeded: !OUTPUT_NAME!
    )
)

echo.
goto :eof

:show_help
echo Test script usage:
echo.
echo 1. Test all programs:
echo    test-all.bat
echo.
echo 2. Test single program:
echo    test-all.bat [program_name]
echo    Example: test-all.bat resize-ncnn.exe
echo.
echo 3. View help:
echo    test-all.bat help
echo.
echo Notes:
echo - Input files need to be placed in assets\input directory
echo - Output results are saved in assets\output directory
echo - The script will automatically clean the output directory
echo - You can add new programs and parameter groups in the script
echo.
goto :eof
