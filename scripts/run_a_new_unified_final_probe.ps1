param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [int]$Episodes = 100,

    [int]$SeedBase = 0,

    [ValidateSet("formal", "smoke")]
    [string]$RunStage = "formal",

    [string]$CheckpointStoreRoot = "checkpoint_store",

    [string]$OutputRoot = "experiment_records\final_method\unified_final_probe",

    [string]$RunId = "",

    [string]$ScenarioId = "S0_default_training_matched",

    [int]$Rows = 0,

    [int]$Cols = 0,

    [double]$ObstacleRatio = -1.0,

    [int]$MaxEpisodeSteps = 0,

    [double]$CoverageStopThreshold = -1.0,

    [string[]]$Groups = @(),

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
    "--run-stage", $RunStage,
    "--checkpoint-store-root", $CheckpointStoreRoot,
    "--output-root", $OutputRoot,
    "--scenario-id", $ScenarioId
)

if ($SeedBase -gt 0) {
    $pythonArgs += @("--seed-base", [string]$SeedBase)
}

if ($RunId -ne "") {
    $pythonArgs += @("--run-id", $RunId)
}

if ($Rows -gt 0) {
    $pythonArgs += @("--rows", [string]$Rows)
}

if ($Cols -gt 0) {
    $pythonArgs += @("--cols", [string]$Cols)
}

if ($ObstacleRatio -ge 0.0) {
    $pythonArgs += @("--obstacle-ratio", [string]$ObstacleRatio)
}

if ($MaxEpisodeSteps -gt 0) {
    $pythonArgs += @("--max-episode-steps", [string]$MaxEpisodeSteps)
}

if ($CoverageStopThreshold -ge 0.0) {
    $pythonArgs += @("--coverage-stop-threshold", [string]$CoverageStopThreshold)
}

if ($Groups.Count -gt 0) {
    $groupText = (($Groups | ForEach-Object { $_.Split(",") } | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }) -join ",")
    if ($groupText -ne "") {
        $pythonArgs += @("--groups", $groupText)
    }
}

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

if ($ContinueOnFailure) {
    $pythonArgs += "--continue-on-failure"
}

Write-Host "[A_new_unified_final_probe] Device=$Device RunStage=$RunStage Episodes=$Episodes SeedBase=$SeedBase ScenarioId=$ScenarioId OutputRoot=$OutputRoot CheckpointStoreRoot=$CheckpointStoreRoot DryRun=$($DryRun.IsPresent)"
Write-Host "[A_new_unified_final_probe] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
