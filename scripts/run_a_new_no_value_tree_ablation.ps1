param(
    [ValidateSet("smoke", "pilot", "formal")]
    [string]$RunStage = "smoke",

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "experiments\final_method\run_a_new_no_value_tree_ablation.py",
    "--run-stage", $RunStage,
    "--device", $Device
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

Write-Host "[A_new_D] RunStage=$RunStage Device=$Device DryRun=$($DryRun.IsPresent)"
Write-Host "[A_new_D] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
