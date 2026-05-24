param(
    [ValidateSet("smoke", "pilot", "formal")]
    [string]$RunStage = "smoke",

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [switch]$DryRun,

    [switch]$NoCopyCheckpoints
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "experiments\ablations\run_ablation_batch.py",
    "--ablation-ids", "F6,F7",
    "--run-stage", $RunStage,
    "--device", $Device
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

if ($NoCopyCheckpoints) {
    $pythonArgs += "--no-copy-checkpoints"
}

Write-Host "[f6-f7] RunStage=$RunStage Device=$Device DryRun=$($DryRun.IsPresent) NoCopyCheckpoints=$($NoCopyCheckpoints.IsPresent)"
Write-Host "[f6-f7] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
