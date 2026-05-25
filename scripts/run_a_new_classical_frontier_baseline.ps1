param(
    [ValidateSet("smoke", "pilot", "formal")]
    [string]$RunStage = "smoke",

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cpu",

    [switch]$DryRun,

    [string]$OutputRoot = "outputs",

    [int]$Episodes = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "experiments\final_method\run_a_new_classical_frontier_baseline.py",
    "--run-stage", $RunStage,
    "--device", $Device,
    "--output-root", $OutputRoot
)

if ($Episodes -gt 0) {
    $pythonArgs += @("--episodes", [string]$Episodes)
}

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

Write-Host "[A_new_B] RunStage=$RunStage Device=$Device DryRun=$($DryRun.IsPresent) Episodes=$Episodes"
Write-Host "[A_new_B] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
