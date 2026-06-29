param([string]$Device = 'cuda')
$root = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $root 'results\_k8_R8.log'
$err  = Join-Path $root 'results\_k8_R8.err'

$cmdArgs = @(
    '-m', 'experiments.heteromini_long',
    '--variant', 'sparse',
    '--n_blocks', '32',
    '--k', '8',
    '--R', '8',
    '--core_mode', 'per_token',
    '--max_steps', '10000',
    '--lr_horizon', '10000',
    '--bs', '16',
    '--seq_len', '128',
    '--seed', '0',
    '--device', $Device
)

$p = Start-Process -FilePath 'python' -ArgumentList $cmdArgs -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru

Write-Output "srcore_b32_k8_R8 @10k gestartet -- PID $($p.Id)"
Write-Output "  stdout -> $out"
Write-Output "  stderr -> $err"
Write-Output "Monitoring: Get-Content '$out' -Wait -Tail 20"
