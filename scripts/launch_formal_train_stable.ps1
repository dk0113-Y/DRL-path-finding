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
    [int]$FixedTrainEpisodeSeedBase = 20259323,
    [int]$FixedFinalProbeSeedBase = 20261323,
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
