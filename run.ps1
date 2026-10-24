param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv --system-site-packages .venv
    } else {
        & python -m venv --system-site-packages .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python environment.' }
}
& $venvPython -c "import sys; exec('try:\n import mujoco, numpy, PIL\nexcept Exception:\n sys.exit(1)')"
if ($LASTEXITCODE -ne 0) {
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
$serverArgs = @('sim/server.py', '--port', "$Port")
if (-not $NoBrowser) { $serverArgs += '--open-browser' }
& $venvPython @serverArgs
if ($LASTEXITCODE -ne 0) { throw 'Simulation stopped with an error. See the output above.' }
