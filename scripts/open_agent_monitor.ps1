$ErrorActionPreference = 'Stop'

# Explorer-launched BAT files do not inherit an activated PowerShell Conda profile.
$taskPythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
$taskCandidates = @(
    $env:XIAOMACHI_MONITOR_PYTHON
    $(if ($taskPythonCommand) { $taskPythonCommand.Source })
    'C:\work\anaconda\python.exe'
    "$env:USERPROFILE\anaconda3\python.exe"
    "$env:USERPROFILE\miniconda3\python.exe"
)
$taskPython = $taskCandidates | Where-Object {
    $_ -and $_ -notlike '*\WindowsApps\*' -and (Test-Path -LiteralPath $_ -PathType Leaf)
} | Select-Object -First 1
if (-not $taskPython) {
    Write-Error 'Python was not found. Set XIAOMACHI_MONITOR_PYTHON to an installed python.exe.'
    exit 1
}
& $taskPython -B -X utf8 (Join-Path $PSScriptRoot 'open_agent_monitor.py') @args
exit $LASTEXITCODE
