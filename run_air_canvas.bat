@echo off
setlocal EnableDelayedExpansion
title Air Canvas Pro [Neural Cybernetic Studio]
color 0B

echo.
echo    ======================================================================
echo          [+] ARTIFICIAL INTELLIGENCE - NEURAL SYSTEMS CANVAS [+]
echo    ======================================================================
echo                     .---------------------------.
echo                    /      .---------------.      \
echo                   ^|      /    .-'''-.      \      ^|
echo                   ^|     ^|    /   / \ \      ^|     ^|
echo                   ^|     ^|   ^|   ^| A ^| ^|     ^|     ^|
echo                   ^|     ^|    \   \ / /      ^|     ^|
echo                   ^|      \    '-...-'      /      ^|
echo                    \      '---------------'      /
echo                     '---------------------------'
echo           CYBERNETIC INTEGRATION  *  3D BLENDER TOOLBOX
echo    ======================================================================
echo.

cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system PATH.
    echo Please install Python 3.10+ and check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo [1/2] Checking Python dependencies...
python -c "import cv2, mediapipe, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages: opencv-python, mediapipe, numpy...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to install dependencies. Check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo Dependencies are ready.
)

echo.
echo [2/2] Launching Air Canvas Application...
echo.
echo ======================================================================
echo   PURE BUTTON CONTROLS (ALL HAND SIGNS REMOVED):
echo   * DRAW: Raise Index finger and draw in air
echo   * BUTTONS: Simply tap any on-screen button with your Index fingertip!
echo     - CLEAR button: Clears canvas
echo     - SAVE button: Exports PNG to saved_canvas folder
echo     - PENCIL SIZE: Tap [-] / [+] or presets S, M, L, XL
echo     - 3D ZOOM: Tap [Z-] or [Z+] on the toolbar
echo     - 3D TOOLS: Tap 3D TUBE, 3D SPHERE, CUBE, CYLINDER, PYRAMID, TORUS
echo   * MOVE / MERGE: Pinch (Thumb + Index) to drag any object
echo   * DUSTBIN: Drag any object into the bottom-right DUSTBIN to DELETE it!
echo.
echo   KEYBOARD SHORTCUTS:
echo   * [C] Clear all objects
echo   * [S] Save image to PNG
echo   * [1], [2], [3], [4] Pencil sizes: S, M, L, XL
echo   * [ [ ] / [ ] ] Zoom 3D object out / in
echo   * [W] Toggle Wireframe vs Solid Shaded mesh
echo   * [Tab] Toggle Camera View vs Pure Black Studio
echo   * [Q] or [Esc] Quit
echo ======================================================================
echo.

python air_canvas.py
set APP_EXIT_CODE=%errorlevel%

if %APP_EXIT_CODE% neq 0 (
    echo.
    echo ======================================================================
    echo [ERROR] Application exited with code %APP_EXIT_CODE%
    echo ======================================================================
    pause
)

endlocal
