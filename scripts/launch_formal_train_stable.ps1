[CmdletBinding()]
param(
    [string]$PythonExecutable = "python",
    [string]$RunName = "formal_stable_seed0",
    [string]$Device = "cuda",
    [int]$Seed = 0,
    [int]$TotalEnvSteps = 500000,
    [int]$EpsilonDecaySteps = 320000,
    [double]$EpsilonEnd = 0.05,
    [int]$MinReplaySize = 8000,
    [int]$FinalGreedyEpisodes = 100,
    [double]$RewardRevisitPenalty = 0.10,
    [double]$RewardTurnPenaltyScale = 0.05,
    [double]$RewardTurnWeight45 = 0.0,
    [double]$RewardTurnWeight90 = 0.3333333333333333,
    [double]$RewardTurnWeight135 = 0.6666666666666666,
    [double]$RewardTurnWeight180 = 1.0,
    [double]$RewardTimeoutPenalty = 8.0,
    [double]$RewardStepPenalty = 0.02,
    [double]$RewardTerminalBonus = 20.0,
    [double]$RewardInfoScale = 3.0,
    [double]$RewardObstacleWeight = 0.25,
    [int]$BatchSize = 128,
    [int]$ReplayCapacity = 100000,
    [int]$NStep = 3,
    [double]$Gamma = 0.99,
    [double]$LearningRate = 0.0001,
    [int]$TargetUpdateInterval = 1000,
    [double]$GradClipNorm = 10.0,
    [int]$CollectStepsPerIter = 16,
    [int]$LearnerUpdatesPerIter = 2,
    [int]$TrainEveryEnvSteps = 16,
    [int]$FixedTrainEpisodeSeedBase = 20259323,
    [int]$FixedFinalProbeSeedBase = 20261323,
    [bool]$TrainSideOnlyTuning = $true,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$trainArgs = @(
    ".\train_q_agent.py",
    "--device", $Device,
    "--run-name", $RunName,
    "--seed", ([string]$Seed),
    "--total-env-steps", ([string]$TotalEnvSteps),
    "--epsilon-decay-steps", ([string]$EpsilonDecaySteps),
    "--epsilon-end", ($EpsilonEnd.ToString($invariantCulture)),
    "--min-replay-size", ([string]$MinReplaySize),
    "--final-greedy-episodes", ([string]$FinalGreedyEpisodes),
    "--reward-revisit-penalty", ($RewardRevisitPenalty.ToString($invariantCulture)),
    "--reward-turn-penalty-scale", ($RewardTurnPenaltyScale.ToString($invariantCulture)),
    "--reward-turn-weight-45", ($RewardTurnWeight45.ToString($invariantCulture)),
    "--reward-turn-weight-90", ($RewardTurnWeight90.ToString($invariantCulture)),
    "--reward-turn-weight-135", ($RewardTurnWeight135.ToString($invariantCulture)),
    "--reward-turn-weight-180", ($RewardTurnWeight180.ToString($invariantCulture)),
    "--reward-timeout-penalty", ($RewardTimeoutPenalty.ToString($invariantCulture)),
    "--reward-step-penalty", ($RewardStepPenalty.ToString($invariantCulture)),
    "--reward-terminal-bonus", ($RewardTerminalBonus.ToString($invariantCulture)),
    "--reward-info-scale", ($RewardInfoScale.ToString($invariantCulture)),
    "--reward-obstacle-weight", ($RewardObstacleWeight.ToString($invariantCulture)),
    "--batch-size", ([string]$BatchSize),
    "--replay-capacity", ([string]$ReplayCapacity),
    "--n-step", ([string]$NStep),
    "--gamma", ($Gamma.ToString($invariantCulture)),
    "--learning-rate", ($LearningRate.ToString($invariantCulture)),
    "--target-update-interval", ([string]$TargetUpdateInterval),
    "--grad-clip-norm", ($GradClipNorm.ToString($invariantCulture)),
    "--collect-steps-per-iter", ([string]$CollectStepsPerIter),
    "--learner-updates-per-iter", ([string]$LearnerUpdatesPerIter),
    "--train-every-env-steps", ([string]$TrainEveryEnvSteps),
    "--fixed-train-episode-seed-base", ([string]$FixedTrainEpisodeSeedBase),
    "--fixed-final-probe-seed-base", ([string]$FixedFinalProbeSeedBase),
    "--strict-reproducibility",
    "--no-deterministic-warn-only",
    "--no-enable-tf32",
    "--no-enable-cudnn-benchmark",
    "--no-enable-amp",
    "--no-enable-inference-amp",
    "--no-enable-torch-compile",
    "--no-enable-channels-last",
    "--use-fixed-train-episode-seeds",
    "--use-fixed-eval-seeds"
)

if ($TrainSideOnlyTuning) {
    $trainArgs += "--train-side-only-tuning"
}

function Quote-CommandPart {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

Write-Host "PYTHONHASHSEED=$env:PYTHONHASHSEED"
Write-Host "CUBLAS_WORKSPACE_CONFIG=$env:CUBLAS_WORKSPACE_CONFIG"
Write-Host "Repository root: $repoRoot"
Write-Host "Run name: $RunName"
Write-Host "Command:"
Write-Host ("  " + ((@($PythonExecutable) + $trainArgs | ForEach-Object { Quote-CommandPart $_ }) -join " "))

if ($DryRun) {
    Write-Host "Dry run only; Python was not launched."
    exit 0
}

Push-Location $repoRoot
try {
    & $PythonExecutable @trainArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
