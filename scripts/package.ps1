param([string]$PythonPath = "")
$ErrorActionPreference = "Stop"
$repoDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoDirectory
if (-not $PythonPath) { $PythonPath = Join-Path $repoDirectory ".venv\Scripts\python.exe" }
& $PythonPath -m pip install -e ".[package]"
if ($LASTEXITCODE -ne 0) { throw "Packaging dependency installation failed." }
& $PythonPath -m PyInstaller --noconfirm --clean --onedir --windowed --name SemanticImageStream `
    --collect-all faster_whisper --collect-all ctranslate2 --collect-all av --collect-all tokenizers `
    --collect-submodules semantic_image_stream --exclude-module PySide6.QtWebEngineCore `
    --exclude-module PySide6.QtWebEngineWidgets --exclude-module PySide6.QtWebEngineQuick `
    --add-data "LICENSE;." --add-data "docs/third-party.md;." scripts/launcher.py
if ($LASTEXITCODE -ne 0) { throw "Packaging failed." }
Write-Host "Build: dist\SemanticImageStream\SemanticImageStream.exe"
Write-Host "Models download on first use. Optional CLIP/spaCy packages must be installed before building if needed."
