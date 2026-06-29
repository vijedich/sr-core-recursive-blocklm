param([string]$Device = 'cuda')
$root = Split-Path -Parent $PSScriptRoot

foreach ($seed in @(1, 2)) {
    $out = Join-Path $root "results\_k8_R6_seed${seed}.log"
    $err = Join-Path $root "results\_k8_R6_seed${seed}.err"

    $args = @(
        '-m', 'experiments.heteromini_long',
        '--variant',    'sparse',
        '--n_blocks',   '32',
        '--k',          '8',
        '--R',          '6',
        '--core_mode',  'per_token',
        '--max_steps',  '15000',
        '--lr_horizon', '15000',
        '--bs',         '16',
        '--seq_len',    '128',
        '--seed',       "$seed",
        '--device',     $Device
    )

    $p = Start-Process python -ArgumentList $args `
        -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru
    Write-Output "srcore_b32_k8_R6 seed=$seed gestartet -- PID $($p.Id) -> $out"
    $p.WaitForExit()
    Write-Output "seed=$seed fertig (Exit $($p.ExitCode))"
}

Write-Output "=== BEIDE SEEDS FERTIG ==="
