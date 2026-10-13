param(
    [string]$PythonPath = "",
    [switch]$WithClip,
    [switch]$WithSemantic,
    [switch]$WithDev
)
$ErrorActionPreference = "Stop"
$repoDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoDirectory
if ($PythonPath) {
    & $PythonPath -m venv .venv
} else {
    $created = $false
    foreach ($version in @("3.11", "3.12", "3.13")) {
        & py "-$version" -m venv .venv
        if ($LASTEXITCODE -eq 0) { $created = $true; break }
    }
    if (-not $created) { throw "Install Python 3.11-3.13 or pass -PythonPath with a compatible python.exe." }
}
if ($LASTEXITCODE -ne 0) { throw "Could not create the isolated Python environment." }
$appPython = Join-Path $repoDirectory ".venv\Scripts\python.exe"
& $appPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
$extras = @("search")
if ($WithClip) { $extras += "clip" }
if ($WithSemantic) { $extras += "semantic" }
if ($WithDev) { $extras += "dev" }
$installTarget = ".[" + ($extras -join ',') + "]"
& $appPython -m pip install -e $installTarget
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
if ($WithSemantic) {
    & $appPython -m spacy download en_core_web_sm
    if ($LASTEXITCODE -ne 0) { throw "Could not install the optional English semantic model." }
}
if (-not (Test-Path -LiteralPath "config.local.toml")) {
    Copy-Item -LiteralPath "config.example.toml" -Destination "config.local.toml"
}
Write-Host "Ready. Launch: .\.venv\Scripts\python.exe -m semantic_image_stream --config config.local.toml"
