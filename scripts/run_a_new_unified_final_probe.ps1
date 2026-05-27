param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [int]$Episodes = 100,

    [int]$SeedBase = 0,

    [string]$CheckpointStoreRoot = "checkpoint_store",

    [string]$OutputRoot = "experiment_records\final_method\unified_final_probe",

    [string]$RunId = "",

    [switch]$DryRun,

    [switch]$ContinueOnFailure
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "experiments\final_method\run_a_new_unified_final_probe.py",
    "--device", $Device,
    "--episodes", [string]$Episodes,
    "--checkpoint-store-root", $CheckpointStoreRoot,
    "--output-root", $OutputRoot
)

if ($SeedBase -gt 0) {
    $pythonArgs += @("--seed-base", [string]$SeedBase)
}

if ($RunId -ne "") {
    $pythonArgs += @("--run-id", $RunId)
}

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

if ($ContinueOnFailure) {
    $pythonArgs += "--continue-on-failure"
}

Write-Host "[A_new_unified_final_probe] Device=$Device Episodes=$Episodes SeedBase=$SeedBase OutputRoot=$OutputRoot CheckpointStoreRoot=$CheckpointStoreRoot DryRun=$($DryRun.IsPresent)"
Write-Host "[A_new_unified_final_probe] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
