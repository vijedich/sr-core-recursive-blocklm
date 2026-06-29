param([string]$Device = 'cuda')
$root = Split-Path -Parent $PSScriptRoot
$ck   = Join-Path $root 'results\hm_cont_hm_srcore_b32_k8_R8_s0.pt'

# 1) heteromini_eval
$out1 = Join-Path $root 'results\_k8_R8_hm_eval.log'
$err1 = Join-Path $root 'results\_k8_R8_hm_eval.err'
$p1 = Start-Process python @('-m','experiments.heteromini_eval',
    '--checkpoint', $ck, '--device', $Device, '--n_batches', '40') `
    -WorkingDirectory $root -RedirectStandardOutput $out1 `
    -RedirectStandardError $err1 -WindowStyle Hidden -PassThru
Write-Output "heteromini_eval PID $($p1.Id) -> $out1"
$p1.WaitForExit()
Write-Output "heteromini_eval fertig (Exit $($p1.ExitCode))"

# 2) gain_seen_unknown
$out2 = Join-Path $root 'results\_k8_R8_gain_su.log'
$err2 = Join-Path $root 'results\_k8_R8_gain_su.err'
$p2 = Start-Process python @('-m','experiments.gain_analysis',
    '--checkpoints', $ck, '--seen_unknown',
    '--device', $Device, '--n_batches', '40') `
    -WorkingDirectory $root -RedirectStandardOutput $out2 `
    -RedirectStandardError $err2 -WindowStyle Hidden -PassThru
Write-Output "gain_seen_unknown PID $($p2.Id) -> $out2"
$p2.WaitForExit()
Write-Output "gain_seen_unknown fertig (Exit $($p2.ExitCode))"

# 3) anytime_inference R1..R8
$out3 = Join-Path $root 'results\_k8_R8_anytime.log'
$err3 = Join-Path $root 'results\_k8_R8_anytime.err'
$p3 = Start-Process python @('scripts/anytime_inference.py',
    '--ckpt', $ck, '--device', $Device,
    '--n_batches', '40', '--r_list', '1','2','3','4','6','8') `
    -WorkingDirectory $root -RedirectStandardOutput $out3 `
    -RedirectStandardError $err3 -WindowStyle Hidden -PassThru
Write-Output "anytime_inference PID $($p3.Id) -> $out3"
$p3.WaitForExit()
Write-Output "anytime_inference fertig (Exit $($p3.ExitCode))"

Write-Output "=== ALLE EVAL-SCHRITTE FERTIG ==="
