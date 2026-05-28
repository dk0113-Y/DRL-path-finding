param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [int]$Episodes = 100,

    [string]$CheckpointDir = "outputs\AN_tuned_v1_final_4ch_no_frontier_raster_formal_20260525_171957\checkpoints",

    [string]$OutputRoot = "experiment_records\final_method\a_checkpoint_s2_s3_probe",

    [string[]]$Scenarios = @("S2", "S3"),

    [string[]]$CheckpointVariants = @("best", "560000", "580000"),

    [switch]$DryRun,

    [switch]$Smoke,

    [switch]$ContinueOnFailure
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

function Resolve-RepoPathText {
    param([string]$PathText)

    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return [System.IO.Path]::GetFullPath($PathText)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PathText))
}

function Get-AvailableCheckpointList {
    param([string]$Root)

    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        return @()
    }
    return @(
        Get-ChildItem -LiteralPath $Root -Recurse -File -Filter "*.pt" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
}

function Format-AvailableCheckpointList {
    param([string]$Root)

    $available = Get-AvailableCheckpointList $Root
    if ($available.Count -eq 0) {
        return "No .pt files found under $Root"
    }
    return ($available -join [Environment]::NewLine)
}

function Resolve-CheckpointVariant {
    param(
        [string]$Variant,
        [string]$Root
    )

    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        throw "CheckpointDir does not exist: $Root"
    }

    $normalized = $Variant.ToLowerInvariant()
    if ($normalized -eq "best") {
        $candidate = Join-Path $Root "best.pt"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
        throw "Checkpoint variant '$Variant' not found at expected path: $candidate`nAvailable .pt files:`n$(Format-AvailableCheckpointList $Root)"
    }

    if ($normalized -ne "560000" -and $normalized -ne "580000") {
        throw "Unsupported checkpoint variant '$Variant'. Supported variants: best, 560000, 580000."
    }

    $padded = "{0:D9}" -f [int]$normalized
    $priorityCandidates = @(
        (Join-Path $Root "ckpt_step_${normalized}.pt"),
        (Join-Path (Join-Path $Root "model_select") "env_${padded}.pt"),
        (Join-Path $Root "env_${padded}.pt")
    )
    foreach ($candidate in $priorityCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $matches = @(
        Get-ChildItem -LiteralPath $Root -Recurse -File -Filter "*.pt" |
            Where-Object { $_.Name -like "*${normalized}*" } |
            Sort-Object FullName
    )
    if ($matches.Count -eq 1) {
        return $matches[0].FullName
    }
    if ($matches.Count -gt 1) {
        $paths = ($matches | ForEach-Object { $_.FullName }) -join [Environment]::NewLine
        throw "Checkpoint variant '$Variant' matched multiple candidates. Refusing to guess:`n$paths"
    }

    throw "Checkpoint variant '$Variant' not found under $Root.`nAvailable .pt files:`n$(Format-AvailableCheckpointList $Root)"
}

function Quote-CommandArg {
    param([string]$Value)

    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Format-PythonCommand {
    param([object[]]$CommandArgs)

    return "python " + (($CommandArgs | ForEach-Object { Quote-CommandArg ([string]$_) }) -join " ")
}

$scenarioPath = Join-Path $repoRoot "experiments\final_method\environment_shift_scenarios.json"
$scenarioConfig = Get-Content -Raw $scenarioPath | ConvertFrom-Json

$scenarioByKey = @{}
foreach ($scenario in $scenarioConfig.scenarios) {
    $scenarioKey = ([string]$scenario.scenario_key).ToUpperInvariant()
    $scenarioIdKey = ([string]$scenario.scenario_id).ToUpperInvariant()
    if ($scenarioKey -eq "S2" -or $scenarioKey -eq "S3") {
        $scenarioByKey[$scenarioKey] = $scenario
        $scenarioByKey[$scenarioIdKey] = $scenario
    }
}

$requestedScenarios = Resolve-TokenList $Scenarios
$requestedVariants = Resolve-TokenList $CheckpointVariants
if ($requestedScenarios.Count -eq 0) {
    throw "At least one scenario must be requested."
}
if ($requestedVariants.Count -eq 0) {
    throw "At least one checkpoint variant must be requested."
}

$checkpointRoot = Resolve-RepoPathText $CheckpointDir
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
$failures = @()
$planned = 0

Write-Host "[a_checkpoint_s2_s3_probe] Device=$Device RunStage=$runStage Episodes=$effectiveEpisodes OutputRoot=$effectiveOutputRoot CheckpointDir=$checkpointRoot DryRun=$($DryRun.IsPresent) Smoke=$($Smoke.IsPresent)"

foreach ($variant in $requestedVariants) {
    try {
        $checkpointPath = Resolve-CheckpointVariant -Variant $variant -Root $checkpointRoot
    }
    catch {
        $failure = "checkpoint_variant=${variant}: $($_.Exception.Message)"
        $failures += $failure
        Write-Host "[a_checkpoint_s2_s3_probe] failed $failure"
        if (-not $ContinueOnFailure) {
            throw
        }
        continue
    }

    foreach ($scenarioToken in $requestedScenarios) {
        $lookupKey = $scenarioToken.ToUpperInvariant()
        if (-not $scenarioByKey.ContainsKey($lookupKey)) {
            throw "Unsupported scenario '$scenarioToken'. Use S2, S3, or the full scenario_id."
        }

        $scenario = $scenarioByKey[$lookupKey]
        $scenarioId = [string]$scenario.scenario_id
        $runId = "A_checkpoint_${variant}_${scenarioId}_${timestamp}"
        if ($Smoke) {
            $runId = "A_checkpoint_${variant}_${scenarioId}_smoke_${timestamp}"
        }

        $pythonArgs = @(
            "experiments\final_method\run_a_new_unified_final_probe.py",
            "--groups", "A",
            "--a-checkpoint-path", $checkpointPath,
            "--checkpoint-variant", $variant,
            "--scenario-id", $scenarioId,
            "--rows", [string]$scenario.rows,
            "--cols", [string]$scenario.cols,
            "--obstacle-ratio", [string]$scenario.obstacle_ratio,
            "--max-episode-steps", [string]$scenario.max_episode_steps,
            "--coverage-stop-threshold", [string]$scenario.coverage_stop_threshold,
            "--seed-base", [string]$scenario.seed_base,
            "--episodes", [string]$effectiveEpisodes,
            "--run-stage", $runStage,
            "--device", $Device,
            "--output-root", $effectiveOutputRoot,
            "--run-id", $runId
        )

        $planned += 1
        Write-Host "[a_checkpoint_s2_s3_probe] checkpoint_variant=$variant"
        Write-Host "[a_checkpoint_s2_s3_probe] checkpoint_path=$checkpointPath"
        Write-Host "[a_checkpoint_s2_s3_probe] scenario_id=$scenarioId"
        Write-Host "[a_checkpoint_s2_s3_probe] episodes=$effectiveEpisodes"
        Write-Host "[a_checkpoint_s2_s3_probe] seed_base=$($scenario.seed_base)"
        Write-Host "[a_checkpoint_s2_s3_probe] output_root=$effectiveOutputRoot"
        Write-Host "[a_checkpoint_s2_s3_probe] run_id=$runId"
        Write-Host "[a_checkpoint_s2_s3_probe] command: $(Format-PythonCommand -CommandArgs $pythonArgs)"

        if ($DryRun) {
            continue
        }

        & python @pythonArgs
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            $failure = "checkpoint_variant=${variant} scenario=${scenarioId}: exit ${exitCode}"
            $failures += $failure
            Write-Host "[a_checkpoint_s2_s3_probe] failed $failure"
            if (-not $ContinueOnFailure) {
                exit $exitCode
            }
        }
    }
}

if ($DryRun) {
    Write-Host "[a_checkpoint_s2_s3_probe] dry-run planned_tasks=$planned"
}

if ($failures.Count -gt 0) {
    Write-Host "[a_checkpoint_s2_s3_probe] failures: $($failures -join '; ')"
    exit 1
}

Write-Host "[a_checkpoint_s2_s3_probe] completed planned_tasks=$planned"
exit 0
