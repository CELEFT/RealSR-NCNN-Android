@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "ROOT_DIR=%SCRIPT_DIR%\.."
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "WIN_X64_DIR=%ROOT_DIR%\Win-x64"
set "INPUT_DIR=%ROOT_DIR%\input"
set "OUTPUT_DIR=%ROOT_DIR%\output"
set "REPORT_DIR=%ROOT_DIR%\report"

set total_param_groups=0
set total_output_files=0
set total_passed_tests=0
set total_failed_tests=0

REM Create report directory
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

REM Initialize CSV file with timestamp (using PowerShell for 24-hour format)
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do (
    set "timestamp=%%a"
)
REM Get target program name for CSV filename
if "%1"=="" (
    set "csv_prog=all"
) else (
    set "csv_prog=%~1"
    set "csv_prog=!csv_prog:-ncnn.exe=!"
)
set "CSV_FILE=%REPORT_DIR%\test_!csv_prog!_!timestamp!.csv"

REM Write CSV header
echo input_filename,program_name,param_group,params,output_filename>"%CSV_FILE%"

if not exist "%WIN_X64_DIR%" (
    echo Error: Win-x64 directory not found at: %WIN_X64_DIR%
    goto endscript
)

if not exist "%INPUT_DIR%" (
    echo Error: Input directory not found at: %INPUT_DIR%
    goto endscript
)

echo Cleaning output directory...
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if exist "%OUTPUT_DIR%\*" del /f /q "%OUTPUT_DIR%\*" >nul 2>&1

set file_count=0
for %%f in ("%INPUT_DIR%\*") do set /a file_count+=1
if !file_count! equ 0 (
    echo No files in input directory %INPUT_DIR%
    goto endscript
)

REM Parse arguments
set "target_program="
set "generate_html=1"

:parse_args
if "%1"=="" goto args_done
if /i "%1"=="help" goto showhelp
if /i "%1"=="--help" goto showhelp
if /i "%1"=="-h" goto showhelp
if /i "%1"=="--no-html" (
    set "generate_html=0"
    shift
    goto parse_args
)
REM Check if argument is a valid program name
if /i "%1"=="resize-ncnn.exe" (
    set "target_program=%~1"
    shift
    goto parse_args
)
if /i "%1"=="realcugan-ncnn.exe" (
    set "target_program=%~1"
    shift
    goto parse_args
)
if /i "%1"=="realsr-ncnn.exe" (
    set "target_program=%~1"
    shift
    goto parse_args
)
if /i "%1"=="srmd-ncnn.exe" (
    set "target_program=%~1"
    shift
    goto parse_args
)
if /i "%1"=="waifu2x-ncnn.exe" (
    set "target_program=%~1"
    shift
    goto parse_args
)
if /i "%1"=="mnnsr-ncnn.exe" (
    set "target_program=%~1"
    shift
    goto parse_args
)
REM Unknown argument, skip it
shift
goto parse_args

:args_done
if "%target_program%"=="" goto test_all

REM Program name already validated in parse_args, now jump to appropriate section
if /i "%target_program%"=="resize-ncnn.exe" goto test_resize_only
if /i "%target_program%"=="realcugan-ncnn.exe" goto test_realcugan_only
if /i "%target_program%"=="realsr-ncnn.exe" goto test_realsr_only
if /i "%target_program%"=="srmd-ncnn.exe" goto test_srmd_only
if /i "%target_program%"=="waifu2x-ncnn.exe" goto test_waifu2x_only
if /i "%target_program%"=="mnnsr-ncnn.exe" goto test_mnnsr_only

echo Program not found: %target_program%
echo Please check the program name
goto endscript

:test_all
echo Testing all programs...
echo.
call :block_resize
call :block_realcugan
call :block_realsr_ncnn
call :block_srmd
call :block_waifu2x
call :block_mnnsr
goto summary

:test_resize_only
echo Testing single program: resize-ncnn.exe
echo.
call :block_resize
goto summary

:test_realcugan_only
echo Testing single program: realcugan-ncnn.exe
echo.
call :block_realcugan
goto summary

:test_realsr_only
echo Testing single program: realsr-ncnn.exe
echo.
call :block_realsr_ncnn
goto summary

:test_srmd_only
echo Testing single program: srmd-ncnn.exe
echo.
call :block_srmd
goto summary

:test_mnnsr_only
echo Testing single program: mnnsr-ncnn.exe
echo.
call :block_mnnsr
goto summary

:test_waifu2x_only
echo Testing single program: waifu2x-ncnn.exe
echo.
call :block_waifu2x
goto summary

:run_test
set "rt_program=%~1"
set "rt_params=%~2"
set "rt_index=%~3"
set "test_passed=0"

echo Test parameter group %rt_index%: %rt_params%
set /a total_param_groups+=1

REM Extract program short name (remove -ncnn.exe suffix)
set "prog_short=!rt_program:-ncnn.exe=!"

for %%i in ("%INPUT_DIR%\*") do (
    set "input_file=%%i"
    set "input_name=%%~ni"
    set "input_ext=%%~xi"

    set "model_name="
    set "scale_suffix="
    set "denoise_suffix="
    set "prev_token="
    
    for %%A in (%rt_params%) do (
        set "token=%%~A"
        
        if /i "!token!"=="-m" (
             REM Remember -m for model name
             set "prev_token=m"
        ) else if /i "!token!"=="-s" (
             REM Remember -s for scale value
             set "prev_token=s"
        ) else if /i "!token!"=="-n" (
             REM Remember -n for denoise value
             set "prev_token=n"
        ) else (
             REM Handle Values - merge with previous parameter
             set "fname=!token!"
             REM Replace forward slash with backslash for consistent path handling
             set "fname=!fname:/=\!"
             if "!fname:\=!" neq "!fname!" (
                 REM Contains path separator, extract filename only
                 for %%F in ("!fname!") do set "fname=%%~nxF" 2>nul
             )
             if "!prev_token!"=="s" (
                 REM -s parameter uses x prefix
                 set "scale_suffix=_x!fname!"
             ) else if "!prev_token!"=="n" (
                 REM -n parameter uses n prefix
                 set "denoise_suffix=_n!fname!"
             ) else if "!prev_token!"=="m" (
                 REM -m parameter value (model name) - use as base name
                 set "model_name=!fname!"
             )
             set "prev_token="
        )
    )

    set "output_name=!prog_short!_!model_name!!scale_suffix!!denoise_suffix!_!input_name!!input_ext!"
    set "output_file=%OUTPUT_DIR%\!output_name!"

    if exist "!output_file!" (
        del /f /q "!output_file!" >nul 2>&1
    )

    echo Processing file: !input_name!!input_ext!
    
    "%WIN_X64_DIR%\!rt_program!" %rt_params% -i "!input_file!" -o "!output_file!"

    if exist "!output_file!" (
        for %%f in ("!output_file!") do set "fsize=%%~zf"
        if !fsize! gtr 0 (
            set test_passed=1
            powershell -NoProfile -Command "Write-Host ([char]0x2705 + ' [OK] Test succeeded: ' + '!output_name!') -ForegroundColor Green"
            REM Write to CSV: input_filename,program_name,param_group,params,output_filename
            echo !input_name!!input_ext!,!prog_short!,!rt_index!,!rt_params!,!output_name!>>"%CSV_FILE%"
        ) else (
            powershell -NoProfile -Command "Write-Host ([char]0x274C + ' [FAIL] Empty output: ' + '!output_name!') -ForegroundColor Red"
            REM Write to CSV: input_filename,program_name,param_group,params,(empty)
            echo !input_name!!input_ext!,!prog_short!,!rt_index!,!rt_params!,>>"%CSV_FILE%"
        )
    ) else (
        powershell -NoProfile -Command "Write-Host ([char]0x274C + ' [FAIL] No output generated: ' + '!output_name!') -ForegroundColor Red"
        REM Write to CSV: input_filename,program_name,param_group,params,(empty)
        echo !input_name!!input_ext!,!prog_short!,!rt_index!,!rt_params!,>>"%CSV_FILE%"
    )
)

echo.

if !test_passed! equ 1 (
    set /a total_passed_tests+=1
) else (
    set /a total_failed_tests+=1
)
exit /b

:block_resize
set "prog_name=resize-ncnn.exe"
if not exist "%WIN_X64_DIR%\%prog_name%" (
    echo Program %prog_name% does not exist, skipping.
    exit /b
)
echo Testing program: %prog_name%
call :run_test "%prog_name%" "-m bicubic -s 2" 1
call :run_test "%prog_name%" "-m bilinear -s 2" 2
call :run_test "%prog_name%" "-m nearest -s 2" 3
call :run_test "%prog_name%" "-m avir -s 2" 4
call :run_test "%prog_name%" "-m avir-lancir -s 2" 5
call :run_test "%prog_name%" "-m avir -s 0.5" 6
call :run_test "%prog_name%" "-m de-nearest -s 2" 7
call :run_test "%prog_name%" "-m de-nearest2 -s 2" 8
call :run_test "%prog_name%" "-m de-nearest3 -s 2" 9
call :run_test "%prog_name%" "-m perfectpixel -s 0" 10
echo.
exit /b

:block_realcugan
set "prog_name=realcugan-ncnn.exe"
if not exist "%WIN_X64_DIR%\%prog_name%" (
    echo Program %prog_name% does not exist, skipping.
    exit /b
)
echo Testing program: %prog_name%
call :run_test "%prog_name%" "-m models-nose -s 2 -n 0" 1
call :run_test "%prog_name%" "-m models-se -s 2 -n -1" 2
call :run_test "%prog_name%" "-m models-se -s 2 -n 0" 3
call :run_test "%prog_name%" "-m models-pro -s 2 -n -1" 4
call :run_test "%prog_name%" "-m models-pro -s 2 -n 0" 5
echo.
exit /b

:block_realsr_ncnn
set "prog_name=realsr-ncnn.exe"
if not exist "%WIN_X64_DIR%\%prog_name%" (
    echo Program %prog_name% does not exist, skipping.
    exit /b
)
echo Testing program: %prog_name%
call :run_test "%prog_name%" "-m models-Real-ESRGAN-anime" 1
call :run_test "%prog_name%" "-m models-Real-ESRGANv3-general -s 4" 2
call :run_test "%prog_name%" "-m models-Real-ESRGANv3-anime -s 3" 3
echo.
exit /b

:block_srmd
set "prog_name=srmd-ncnn.exe"
if not exist "%WIN_X64_DIR%\%prog_name%" (
    echo Program %prog_name% does not exist, skipping.
    exit /b
)
echo Testing program: %prog_name%
call :run_test "%prog_name%" "-s 2" 1
call :run_test "%prog_name%" "-s 3" 2
call :run_test "%prog_name%" "-s 4" 3
echo.
exit /b

:block_waifu2x
set "prog_name=waifu2x-ncnn.exe"
if not exist "%WIN_X64_DIR%\%prog_name%" (
    echo Program %prog_name% does not exist, skipping.
    exit /b
)
echo Testing program: %prog_name%
call :run_test "%prog_name%" "-m models-cunet -n 0" 1
call :run_test "%prog_name%" "-m models-cunet -n 2" 2
call :run_test "%prog_name%" "-m models-upconv_7_anime_style_art_rgb -n 0" 3
call :run_test "%prog_name%" "-m models-upconv_7_anime_style_art_rgb -n 2" 4
call :run_test "%prog_name%" "-m models-upconv_7_photo -n 0" 5
call :run_test "%prog_name%" "-m models-upconv_7_photo -n 2" 6
echo.
exit /b

:block_mnnsr
set "prog_name=mnnsr-ncnn.exe"
if not exist "%WIN_X64_DIR%\%prog_name%" (
    echo Program %prog_name% does not exist, skipping.
    exit /b
)
echo Testing program: %prog_name%
call :run_test "%prog_name%" "-m models-MNN/ESRGAN-MoeSR-jp_Illustration-x4.mnn -s 4" 1
call :run_test "%prog_name%" "-m models-MNN/ESRGAN-Nomos8kSC-x4.mnn -s 4" 2
echo.
exit /b

:summary
set total_output_files=0
for %%f in ("%OUTPUT_DIR%\*") do set /a total_output_files+=1

echo All tests completed!
echo Results saved in %OUTPUT_DIR%
echo.
echo Test Statistics:
powershell -NoProfile -Command "Write-Host ('Total parameter groups tested: ' + $env:total_param_groups) -ForegroundColor Cyan"
powershell -NoProfile -Command "Write-Host ('Total output files generated: ' + $env:total_output_files) -ForegroundColor Yellow"
powershell -NoProfile -Command "Write-Host ([char]0x2705 + ' Total passed tests: ' + $env:total_passed_tests) -ForegroundColor Green"
powershell -NoProfile -Command "Write-Host ([char]0x274C + ' Total failed tests: ' + $env:total_failed_tests) -ForegroundColor Red"

REM Generate HTML report if enabled
if !generate_html! equ 1 (
    echo.
    echo Generating HTML report...
    python "%SCRIPT_DIR%\evaluate_image_consistency.py" "!CSV_FILE!"
    
    REM Open HTML report
    for %%f in ("!CSV_FILE!") do (
        set "html_file=%%~dpnf.html"
        if exist "!html_file!" (
            echo Opening HTML report: !html_file!
            start "" "!html_file!"
        )
    )
)

goto endscript

:showhelp
echo Usage:
echo   test-all.bat [options] [program_name]
echo.
echo Options:
echo   --no-html    Do not generate and open HTML report
echo   -h, --help   Show this help message
echo.
echo Examples:
echo   test-all.bat                    Test all programs
echo   test-all.bat resize-ncnn.exe    Test only resize-ncnn.exe
echo   test-all.bat --no-html          Test all programs without HTML report
echo   test-all.bat --no-html waifu2x-ncnn.exe  Test waifu2x without HTML report
goto endscript

:endscript
endlocal
