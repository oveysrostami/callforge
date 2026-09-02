param(
    [string]$Python = "python",
    [string]$Source = "git+https://github.com/oveysrostami/callforge.git",
    [string]$InstallDirectory = ""
)

$ErrorActionPreference = "Stop"

& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "CallForge requires Python 3.11 or newer. Pass -Python with a compatible interpreter."
}

if (-not $InstallDirectory) {
    $InstallDirectory = Join-Path $env:LOCALAPPDATA "CallForge"
}
$EnvironmentDirectory = Join-Path $InstallDirectory "venv"

if (-not (Test-Path (Join-Path $EnvironmentDirectory "Scripts\python.exe"))) {
    & $Python -m venv $EnvironmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not create the CallForge environment." }
}
$EnvironmentPython = Join-Path $EnvironmentDirectory "Scripts\python.exe"
& $EnvironmentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
& $EnvironmentPython -m pip install --upgrade --force-reinstall $Source
if ($LASTEXITCODE -ne 0) { throw "CallForge installation failed." }

& $EnvironmentPython -m callforge setup --yes --force-skill
if ($LASTEXITCODE -ne 0) { throw "CallForge runtime setup did not complete." }

$ScriptsDirectory = Join-Path $EnvironmentDirectory "Scripts"
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$PathEntries = @($UserPath -split ";" | Where-Object { $_ })
if ($PathEntries -notcontains $ScriptsDirectory) {
    $UpdatedPath = (@($PathEntries) + $ScriptsDirectory) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $UpdatedPath, "User")
}
$env:Path = "$ScriptsDirectory;$env:Path"

Write-Output "CallForge installed: $(Join-Path $ScriptsDirectory 'callforge.exe')"
Write-Output "The command is available in this PowerShell session and future terminals."
Write-Output "Next: callforge init C:\path\to\audio"
