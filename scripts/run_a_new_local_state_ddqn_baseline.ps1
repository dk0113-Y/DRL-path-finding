param(
    [ValidateSet("smoke", "pilot", "formal")]
    [string]$RunStage = "smoke",

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [switch]$DryRun,

    [string]$OutputRoot = "outputs",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Passthrough
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonArgs = @(
    "experiments\final_method\run_a_new_local_state_ddqn_baseline.py",
    "--run-stage", $RunStage,
    "--device", $Device,
    "--output-root", $OutputRoot
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

if ($Passthrough -and $Passthrough.Count -gt 0) {
    $pythonArgs += "--"
    $pythonArgs += $Passthrough
}

Write-Host "[A_new_C] RunStage=$RunStage Device=$Device DryRun=$($DryRun.IsPresent) OutputRoot=$OutputRoot"
Write-Host "[A_new_C] command: python $($pythonArgs -join ' ')"

& python @pythonArgs
exit $LASTEXITCODE
