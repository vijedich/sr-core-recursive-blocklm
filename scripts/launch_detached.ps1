<#
.SYNOPSIS
  Startet einen Befehl als LOSGELOESTEN OS-Prozess — unabhaengig vom Claude-Harness.

.WARUM
  Lange GPU-Laeufe als Harness-Background-Task werden offenbar nach einiger Zeit /
  an Turn-Grenzen gereapt (ganzer Prozessbaum gekillt, kein Crash im Event-Log,
  0-Byte-Task-Output). Per Start-Process gestartete Prozesse sind KEINE Harness-Tasks
  und laufen weiter. DEFAULT fuer alles >= wenige Minuten GPU-Zeit.

.NUTZUNG
  .\scripts\launch_detached.ps1 -LogName seed4 -CmdArgs @(
      '-m','experiments.tinystories_exp','--diverse','--diverse_from_iter','2',
      '--steps','3000','--seed','4','--device','cuda')

  Monitoring danach ueber results\_seed4.log (KEINE Harness-Benachrichtigung!).
#>
param(
    [Parameter(Mandatory=$true)][string]$LogName,
    [Parameter(Mandatory=$true)][string[]]$CmdArgs,
    [string]$Exe = 'python'
)
$root = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $root "results\_$LogName.log"
$err  = Join-Path $root "results\_$LogName.err"
$p = Start-Process -FilePath $Exe -ArgumentList $CmdArgs -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru
Write-Output "Detached PID $($p.Id)"
Write-Output "  stdout -> $out"
Write-Output "  stderr -> $err"
