@echo off
echo ========================================================
echo   DeepSceneLoc -- ViT-B/16 Training Launcher
echo ========================================================
echo.
echo Starting ViT-B/16 Training in the background...
start /B venv\Scripts\python.exe -u run_training_advanced.py --model vit_b16 --data data/processed/places365_mit_full_2026_03_15 --epochs 40 --workers 4 --patience 10 > training.log 2>&1

echo Waiting a few seconds for the training log to initialize...
timeout /t 5 /nobreak > nul

echo.
echo Starting live terminal watcher...
venv\Scripts\python.exe watch_training.py
