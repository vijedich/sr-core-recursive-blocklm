param([string]$Device = 'cuda', [int]$MaxNew = 100)
$root = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $root 'results\_generate.log'
$err  = Join-Path $root 'results\_generate.err'

$cmdArgs = @('scripts/generate_suite.py', '--device', $Device, '--max_new', "$MaxNew")

$p = Start-Process -FilePath 'python' -ArgumentList $cmdArgs -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru

Write-Output "generate_suite gestartet -- PID $($p.Id)"
Write-Output "  stdout -> $out"
Write-Output "  Monitoring: Get-Content '$out' -Wait -Tail 20"
