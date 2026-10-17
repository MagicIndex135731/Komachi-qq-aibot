param(
    [ValidateSet('start','status','stop')]
    [string]$Action = 'status'
)
$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'watch_hy2.py'
& python -B -X utf8 $scriptPath $Action
exit $LASTEXITCODE
