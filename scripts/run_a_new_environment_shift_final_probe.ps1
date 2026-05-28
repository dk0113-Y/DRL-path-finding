param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [int]$Episodes = 100,

    [string]$OutputRoot = "experiment_records\final_method\environment_shift_final_probe",

    [string]$CheckpointStoreRoot = "checkpoint_store",

    [switch]$DryRun,

    [switch]$Smoke,

    [switch]$ContinueOnFailure,

    [string[]]$Scenarios = @("S1", "S2", "S3"),

    [string[]]$Groups = @("A", "B", "D", "E", "R")
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Resolve-TokenList {
    param([string[]]$Items)

    $tokens = @()
    foreach ($item in $Items) {
        foreach ($part in ([string]$item).Split(",")) {
            $token = $part.Trim()
            if ($token -ne "") {
                $tokens += $token
            }
        }
    }
    return $tokens
}

$scenarioPath = Join-Path $repoRoot "experiments\final_method\environment_shift_scenarios.json"
$scenarioConfig = Get-Content -Raw $scenarioPath | ConvertFrom-Json

$scenarioByKey = @{}
foreach ($scenario in $scenarioConfig.scenarios) {
    $scenarioKey = ([string]$scenario.scenario_key).ToUpperInvariant()
    $scenarioIdKey = ([string]$scenario.scenario_id).ToUpperInvariant()
    $scenarioByKey[$scenarioKey] = $scenario
    $scenarioByKey[$scenarioIdKey] = $scenario
}

$requestedScenarios = Resolve-TokenList $Scenarios
$requestedGroups = Resolve-TokenList $Groups
if ($requestedScenarios.Count -eq 0) {
    throw "At least one scenario must be requested."
}
if ($requestedGroups.Count -eq 0) {
    throw "At least one group must be requested."
}

$allowedGroups = @("A", "B", "C", "D", "E", "F", "R")
$normalizedGroups = @()
foreach ($group in $requestedGroups) {
    $normalized = $group.ToUpperInvariant()
    if ($allowedGroups -notcontains $normalized) {
        throw "Unsupported group '$group'. Allowed groups: $($allowedGroups -join ',')."
    }
    if ($normalizedGroups -contains $normalized) {
        throw "Duplicate group '$normalized'."
    }
    $normalizedGroups += $normalized
}

$runStage = "formal"
$effectiveEpisodes = $Episodes
$effectiveOutputRoot = $OutputRoot
if ($Smoke) {
    $runStage = "smoke"
    $effectiveEpisodes = 2
    $effectiveOutputRoot = Join-Path $OutputRoot "smoke"
}

if ($effectiveEpisodes -le 0) {
    throw "Episodes must be > 0."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$groupText = $normalizedGroups -join ","
$failures = @()

Write-Host "[environment_shift_final_probe] Device=$Device RunStage=$runStage Episodes=$effectiveEpisodes Groups=$groupText OutputRoot=$effectiveOutputRoot CheckpointStoreRoot=$CheckpointStoreRoot DryRun=$($DryRun.IsPresent) Smoke=$($Smoke.IsPresent)"

foreach ($scenarioToken in $requestedScenarios) {
    $lookupKey = $scenarioToken.ToUpperInvariant()
    if (-not $scenarioByKey.ContainsKey($lookupKey)) {
        throw "Unsupported scenario '$scenarioToken'. Use S1, S2, S3, or the full scenario_id."
    }

    $scenario = $scenarioByKey[$lookupKey]
    $scenarioId = [string]$scenario.scenario_id
    $runId = "environment_shift_${scenarioId}_${timestamp}"
    if ($Smoke) {
        $runId = "environment_shift_${scenarioId}_smoke_${timestamp}"
    }

    $pythonArgs = @(
        "experiments\final_method\run_a_new_unified_final_probe.py",
        "--device", $Device,
        "--episodes", [string]$effectiveEpisodes,
        "--run-stage", $runStage,
        "--checkpoint-store-root", $CheckpointStoreRoot,
        "--output-root", $effectiveOutputRoot,
        "--run-id", $runId,
        "--scenario-id", $scenarioId,
        "--rows", [string]$scenario.rows,
        "--cols", [string]$scenario.cols,
        "--obstacle-ratio", [string]$scenario.obstacle_ratio,
        "--max-episode-steps", [string]$scenario.max_episode_steps,
        "--coverage-stop-threshold", [string]$scenario.coverage_stop_threshold,
        "--groups", $groupText,
        "--seed-base", [string]$scenario.seed_base
    )

    if ($DryRun) {
        $pythonArgs += "--dry-run"
    }
    if ($ContinueOnFailure) {
        $pythonArgs += "--continue-on-failure"
    }

    Write-Host "[environment_shift_final_probe] scenario=$scenarioId rows=$($scenario.rows) cols=$($scenario.cols) obstacle_ratio=$($scenario.obstacle_ratio) max_episode_steps=$($scenario.max_episode_steps) seed_base=$($scenario.seed_base) run_id=$runId"
    Write-Host "[environment_shift_final_probe] command: python $($pythonArgs -join ' ')"

    & python @pythonArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $failure = "${scenarioId}: exit ${exitCode}"
        $failures += $failure
        Write-Host "[environment_shift_final_probe] failed $failure"
        if (-not $ContinueOnFailure) {
            exit $exitCode
        }
    }
}

if ($failures.Count -gt 0) {
    Write-Host "[environment_shift_final_probe] failures: $($failures -join '; ')"
    exit 1
}

Write-Host "[environment_shift_final_probe] completed scenarios=$($requestedScenarios -join ',')"
exit 0
