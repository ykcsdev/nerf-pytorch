@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Set your config directory
set CONFIG_DIR="configsh"

:: Loop through all .txt files in the config folder
for %%f in (%CONFIG_DIR%\*.txt) do (
    echo Running config: %%~nxf
    python run_nerf.py --config %%f
    echo -----------------------------------------------
    echo Finished running %%~nxf
    echo -----------------------------------------------
)

echo All configs completed.
pause
