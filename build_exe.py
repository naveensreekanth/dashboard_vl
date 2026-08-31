import os
import sys
import subprocess

def main():
    print("==================================================")
    print("   Verilumen ATE Intelligence — .EXE Builder      ")
    print("==================================================")
    
    # 1. Install pyinstaller if not already present
    try:
        import PyInstaller
        print("[1/3] PyInstaller is already installed.")
    except ImportError:
        print("[1/3] Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 2. Build single executable launcher
    print("[2/3] Building standalone VerilumenSuite.exe with PyInstaller...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name", "VerilumenSuite",
        "--icon=NONE",
        "run_suite.py"
    ]
    subprocess.check_call(cmd)

    # 3. Create Windows Desktop Shortcut / Quick Batch Launcher
    print("[3/3] Creating Windows Quick-Launch script...")
    batch_content = """@echo off
title Verilumen ATE Intelligence Suite
echo ========================================================
echo   Launching Verilumen ATE Intelligence Suite (Offline)
echo ========================================================
python run_suite.py
pause
"""
    with open("Launch_Verilumen_Suite.bat", "w") as f:
        f.write(batch_content)

    print("\n==================================================")
    print("  Build complete!")
    print("  1. Executable Folder: dist/VerilumenSuite/VerilumenSuite.exe")
    print("  2. One-Click Windows Launcher: Launch_Verilumen_Suite.bat")
    print("==================================================")

if __name__ == "__main__":
    main()
