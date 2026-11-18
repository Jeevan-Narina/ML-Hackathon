param(
    [string]$Archive = "",
    [string]$Raw = "",
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$BackendRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $BackendRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Create .venv and install requirements first; see laptop1_backend/README.md."
}
if (-not $Archive) { $Archive = Join-Path (Split-Path -Parent (Split-Path -Parent $BackendRoot)) "Student folder.zip" }
if (-not $Raw) { $Raw = Join-Path $BackendRoot "data/raw" }
if (-not $Output) { $Output = Join-Path $BackendRoot "outputs/eda" }
$SourceManifest = Join-Path $Output "source_manifest.csv"
& $PythonExe (Join-Path $PSScriptRoot "extract_dataset.py") --archive $Archive --destination $Raw --manifest $SourceManifest
if ($LASTEXITCODE -ne 0) { throw "Dataset extraction failed ($LASTEXITCODE)." }
& $PythonExe (Join-Path $PSScriptRoot "run_eda.py") --raw $Raw --output $Output --manifest $SourceManifest
if ($LASTEXITCODE -ne 0) { throw "EDA failed ($LASTEXITCODE). Inspect the audit outputs." }
& $PythonExe -m pytest $BackendRoot --junitxml (Join-Path $Output "pytest-results.xml") -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed ($LASTEXITCODE)." }
