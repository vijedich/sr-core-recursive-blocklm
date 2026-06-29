param([string]$Device = 'cuda')
$root = Split-Path -Parent $PSScriptRoot

# Warte bis Training-Prozess fertig ist
Write-Output "Warte auf Cross-Seed-Training..."
$trainingLog = Join-Path $root 'results\_crossseed_k8_R6.log'
while (-not (Select-String -Path $trainingLog -Pattern 'BEIDE SEEDS FERTIG' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 30
}
Write-Output "Training fertig. Starte Eval-Pipeline..."

foreach ($seed in @(1, 2)) {
    $ck = Join-Path $root "results\hm_cont_hm_srcore_b32_k8_R6_s${seed}.pt"
    if (-not (Test-Path $ck)) {
        Write-Output "FEHLER: Checkpoint nicht gefunden: $ck"
        continue
    }

    # gain_seen_unknown
    Write-Output "--- seed=${seed} gain_seen_unknown ---"
    $p = Start-Process python `
        -ArgumentList @('-m','experiments.gain_analysis',
            '--checkpoints', "results/hm_cont_hm_srcore_b32_k8_R6_s${seed}.pt",
            '--seen_unknown', '--device', $Device, '--n_batches', '40') `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $root "results\_k8_R6_seed${seed}_gain_su.log") `
        -RedirectStandardError  (Join-Path $root "results\_k8_R6_seed${seed}_gain_su.err") `
        -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    Write-Output "gain_su seed=$seed fertig (Exit $($p.ExitCode))"

    # anytime_inference
    Write-Output "--- seed=${seed} anytime_inference ---"
    $p = Start-Process python `
        -ArgumentList @('scripts/anytime_inference.py',
            '--ckpt', "results/hm_cont_hm_srcore_b32_k8_R6_s${seed}.pt",
            '--device', $Device, '--n_batches', '40',
            '--r_list', '1','2','3','4','6') `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $root "results\_k8_R6_seed${seed}_anytime.log") `
        -RedirectStandardError  (Join-Path $root "results\_k8_R6_seed${seed}_anytime.err") `
        -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    Write-Output "anytime seed=$seed fertig (Exit $($p.ExitCode))"
}

# Full Benchmark mit allen Seeds
Write-Output "--- full_benchmark (alle Seeds) ---"
$p = Start-Process python `
    -ArgumentList @('scripts/full_benchmark.py', '--device', $Device) `
    -WorkingDirectory $root `
    -RedirectStandardOutput (Join-Path $root 'results\_full_benchmark_crossseed.log') `
    -RedirectStandardError  (Join-Path $root 'results\_full_benchmark_crossseed.err') `
    -WindowStyle Hidden -PassThru
$p.WaitForExit()
Write-Output "full_benchmark fertig (Exit $($p.ExitCode))"

Write-Output "=== CROSS-SEED EVAL KOMPLETT ==="
