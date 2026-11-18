param([string]$Partition = "all", [string]$Output = "", [string]$Processed = "")
$ErrorActionPreference = "Stop"
$BackendRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $BackendRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Install the backend virtual environment and requirements first." }
$RunArguments = @((Join-Path $PSScriptRoot "run_step2.py"), "--partition", $Partition)
if ($Output) { $RunArguments += @("--output", $Output) }
if ($Processed) { $RunArguments += @("--processed", $Processed) }
& $PythonExe @RunArguments
if ($LASTEXITCODE -ne 0) { throw "Step 2 processing failed ($LASTEXITCODE)." }
