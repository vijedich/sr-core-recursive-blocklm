param([string]$Device = 'cuda')
$root = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $root 'results\_full_benchmark.log'
$err  = Join-Path $root 'results\_full_benchmark.err'

$cmdArgs = @(
    'scripts/full_benchmark.py',
    '--device', $Device,
    '--n_batches', '40',
    '--bs', '16',
    '--seq_len', '128'
)

$p = Start-Process -FilePath 'python' -ArgumentList $cmdArgs -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru

Write-Output "full_benchmark gestartet -- PID $($p.Id)"
Write-Output "  stdout -> $out"
Write-Output "  stderr -> $err"
Write-Output "Monitoring: Get-Content '$out' -Wait -Tail 20"
