param([string]$Device = 'cuda')
$root = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $root 'results\_anytime.log'
$err  = Join-Path $root 'results\_anytime.err'

$cmdArgs = @('scripts/anytime_inference.py', '--device', $Device,
             '--n_batches', '40', '--bs', '16', '--seq_len', '128',
             '--r_list', '1', '2', '3', '4', '6')

$p = Start-Process -FilePath 'python' -ArgumentList $cmdArgs -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru

Write-Output "anytime_inference gestartet -- PID $($p.Id)"
Write-Output "  stdout -> $out"
Write-Output "  Monitoring: Get-Content '$out' -Wait -Tail 20"
