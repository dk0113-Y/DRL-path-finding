param(
    [ValidateSet("smoke", "pilot", "formal")]
    [string]$RunStage = "smoke",

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [switch]$DryRun,

    [switch]$NoCopyCheckpoints,

    [string]$Only = "R1,R2,R3,R4,R5"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "experiments\final_method\run_a_new_reward_ablation_batch.py",
    "--reward-ablation-ids", $Only,
    "--run-stage", $RunStage,
    "--device", $Device
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

if ($NoCopyCheckpoints) {
    $pythonArgs += "--no-copy-checkpoints"
}

Write-Host "[A_new_R] RunStage=$RunStage Device=$Device DryRun=$($DryRun.IsPresent) NoCopyCheckpoints=$($NoCopyCheckpoints.IsPresent) Only=$Only"
Write-Host "[A_new_R] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
