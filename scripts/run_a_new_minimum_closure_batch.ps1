param(
    [ValidateSet("smoke", "pilot", "formal")]
    [string]$RunStage = "formal",

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [ValidateSet("minimum_closure")]
    [string]$RunSet = "minimum_closure",

    [switch]$DryRun,

    [switch]$IncludeB,

    [switch]$IncludeAllRewardAblations,

    [switch]$ContinueOnFailure,

    [bool]$StopOnFailure = $true,

    [string]$OutputRoot = "outputs",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "experiments\final_method\run_a_new_minimum_closure_batch.py",
    "--run-stage", $RunStage,
    "--device", $Device,
    "--run-set", $RunSet,
    "--output-root", $OutputRoot
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

if ($IncludeB) {
    $pythonArgs += "--include-b"
}

if ($IncludeAllRewardAblations) {
    $pythonArgs += "--include-all-reward-ablations"
}

if ($ContinueOnFailure -or -not $StopOnFailure) {
    $pythonArgs += "--continue-on-failure"
}

if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $pythonArgs += "--extra-args"
    $pythonArgs += $ExtraArgs
}

Write-Host "[A_new_minimum_closure] RunStage=$RunStage Device=$Device RunSet=$RunSet DryRun=$($DryRun.IsPresent) IncludeB=$($IncludeB.IsPresent) IncludeAllRewardAblations=$($IncludeAllRewardAblations.IsPresent) ContinueOnFailure=$($ContinueOnFailure.IsPresent) OutputRoot=$OutputRoot"
if ($IncludeB) {
    Write-Host "[A_new_minimum_closure] IncludeB selected: B is a CPU non-learning benchmark and is launched with device=cpu."
}
Write-Host "[A_new_minimum_closure] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
