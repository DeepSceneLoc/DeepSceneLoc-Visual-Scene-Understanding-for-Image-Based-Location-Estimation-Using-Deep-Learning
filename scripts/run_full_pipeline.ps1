# Full Pipeline: Wait for tar extraction -> Organize -> Split -> Train
# Logs to: logs\places365_pipeline.log and logs\places365_training.log
param(
    [int]$TarPid = 2224,
    [int]$MaxPerClass = 2000,
    [int]$Epochs = 20,
    [int]$Batch = 16,
    [int]$Workers = 2
)

$Root   = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root "venv\Scripts\python.exe"
$Log    = Join-Path $Root "logs\places365_pipeline.log"
$TLog   = Join-Path $Root "logs\places365_training.log"

function Log($msg) {
    $ts = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

Log "=== PIPELINE STARTED (tar PID=$TarPid) ==="

# --- STEP 1: Wait for tar to finish ---
Log "Step 1: Waiting for tar extraction to complete..."
$tarRunning = $true
while ($tarRunning) {
    try {
        $proc = Get-Process -Id $TarPid -ErrorAction Stop
        $sizeMB = [math]::Round(((Get-ChildItem (Join-Path $Root "data\raw") -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum / 1MB), 0)
        Log "  tar still running (CPU=$([math]::Round($proc.CPU,1))s) - extracted ${sizeMB} MB so far"
        Start-Sleep -Seconds 60
    } catch {
        $tarRunning = $false
        Log "  tar process ended."
    }
}

# Final extraction size
$finalGB = [math]::Round(((Get-ChildItem (Join-Path $Root "data\raw") -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB), 3)
Log "Step 1 DONE - Extraction complete. Total: ${finalGB} GB in data\raw"

# --- STEP 2: Final organization pass ---
Log "Step 2: Running final organise_images (max_per_class=$MaxPerClass)..."
$organiseCmd = @"
from pathlib import Path
from scripts.download_dataset import organise_images
organise_images(
    raw_dir=Path('data/raw/train'),
    output_dir=Path('data/processed/places365_raw'),
    categories_file=Path('data/raw/categories_places365.txt'),
    max_per_class=$MaxPerClass
)
"@
Push-Location $Root
& $Python -c $organiseCmd 2>&1 | Tee-Object -FilePath (Join-Path $Root "logs\places365_organize_final.log") | ForEach-Object { Log "  [org] $_" }
$orgExit = $LASTEXITCODE
Pop-Location

if ($orgExit -ne 0) {
    Log "ERROR: organise_images failed (exit $orgExit). Check logs\places365_organize_final.log"
    exit 1
}
Log "Step 2 DONE - Organisation complete."

# Verify class counts
Push-Location $Root
$classCounts = Get-ChildItem "data\processed\places365_raw" -Directory | ForEach-Object {
    $cnt = (Get-ChildItem $_.FullName -File -EA SilentlyContinue | Measure-Object).Count
    "$($_.Name)=$cnt"
}
Pop-Location
Log "  Class counts: $($classCounts -join ', ')"

# --- STEP 3: Split dataset 70/15/15 ---
Log "Step 3: Splitting dataset (70/15/15)..."
Push-Location $Root
& $Python scripts\split_dataset.py `
    --data "data\processed\places365_raw" `
    --out  "data\processed\places365" `
    --copy 2>&1 | Tee-Object -FilePath (Join-Path $Root "logs\places365_split.log") | ForEach-Object { Log "  [split] $_" }
$splitExit = $LASTEXITCODE
Pop-Location

if ($splitExit -ne 0) {
    Log "ERROR: split_dataset.py failed (exit $splitExit). Check logs\places365_split.log"
    exit 1
}
Log "Step 3 DONE - Dataset split complete."

# Verify split counts
Push-Location $Root
foreach ($split in @('train','val','test')) {
    if (Test-Path "data\processed\places365\$split") {
        $cnt = (Get-ChildItem "data\processed\places365\$split" -Recurse -File -EA SilentlyContinue | Measure-Object).Count
        Log "  ${split}: $cnt images"
    }
}
Pop-Location

# --- STEP 4: Train model ---
Log "Step 4: Starting ResNet-50 training ($Epochs epochs, batch=$Batch, workers=$Workers)..."
Log "  Training log -> logs\places365_training.log"
Push-Location $Root
& $Python run_training.py `
    --data    "data\processed\places365" `
    --epochs  $Epochs `
    --batch   $Batch `
    --workers $Workers 2>&1 | Tee-Object -FilePath $TLog | ForEach-Object {
        $_ | Add-Content -Path $Log -Encoding UTF8
        Write-Host $_
    }
$trainExit = $LASTEXITCODE
Pop-Location

if ($trainExit -ne 0) {
    Log "ERROR: Training failed (exit $trainExit). Check logs\places365_training.log"
    exit 1
}

Log "=== PIPELINE COMPLETE - Training finished successfully! ==="
Log "  Results: results\metrics\resnet50_evaluation.json"
Log "  Training curves: logs\training_curves.png"
