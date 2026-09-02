param(
    [string]$Python = "python",
    [string]$EnvironmentDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
if (-not $EnvironmentDirectory) {
    $EnvironmentDirectory = Join-Path $ProjectDirectory ".venv"
}

& $Python -m venv $EnvironmentDirectory
$EnvironmentPython = Join-Path $EnvironmentDirectory "Scripts\python.exe"
& $EnvironmentPython -m pip install --upgrade pip
& $EnvironmentPython -m pip install $ProjectDirectory
$CallForge = Join-Path $EnvironmentDirectory "Scripts\callforge.exe"
& $CallForge setup --yes
Write-Output "CallForge installed at $CallForge"

