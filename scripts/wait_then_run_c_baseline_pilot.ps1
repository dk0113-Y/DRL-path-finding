<#
.SYNOPSIS
Wait for the D structural ablation, then run the C local-state DDQN pilot.

.DESCRIPTION
Detects currently running Python processes for D group structural ablation work:
  python experiments\ablations\run_ablation_batch.py --ablation-ids D --run-stage formal
  python experiments\ablations\run_ablation_batch.py --preset structural_core_batch --run-stage formal
  python experiments\ablations\run_ablation_train.py --ablation-id D_ablation_no_value_tree --run-stage formal

If one or more matching processes are found, waits for those processes to exit.
After D exits, checks the curated D artifacts and then starts:
  python .\experiments\baselines\run_baseline_batch.py --baseline-id C_baseline_local_state_ddqn --run-stage pilot --device <Device> --extra-train-args "..."

If no matching D process is found, asks for explicit confirmation before
launching unless -Force is provided. Use -SkipArtifactCheck only after manually
confirming that skipping the D artifact gate is intentional.

Smoke/pilot runs cannot enter paper Results. This C pilot is only for interface
and short-horizon training validation; final paper evidence requires a formal
run with complete final_probe artifacts.

.EXAMPLE
.\scripts\wait_then_run_c_baseline_pilot.ps1 -DryRun

.EXAMPLE
.\scripts\wait_then_run_c_baseline_pilot.ps1 -Device cuda

.EXAMPLE
.\scripts\wait_then_run_c_baseline_pilot.ps1 -Device cuda -Force

.EXAMPLE
.\scripts\wait_then_run_c_baseline_pilot.ps1 -Device cuda -Force -SkipArtifactCheck
#>

[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [ValidateRange(1, 86400)]
    [int]$PollSeconds = 60,
    [switch]$Force,
    [switch]$DryRun,
    [switch]$SkipArtifactCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent (Resolve-Path -LiteralPath $MyInvocation.MyCommand.Path)
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDir "..")).Path

$pythonProcessNamePattern = "^(python|pythonw|python3|py)(\.exe)?$"
$formalStagePattern = "(^|\s)--run-stage(?:\s+|=)formal(\s|$)"
$batchScriptPattern = "experiments[\\/]+ablations[\\/]+run_ablation_batch\.py|run_ablation_batch\.py"
$trainScriptPattern = "experiments[\\/]+ablations[\\/]+run_ablation_train\.py|run_ablation_train\.py"
$batchDPattern = "(^|\s)--ablation-ids(?:\s+|=)D(\s|$)"
$batchStructuralPattern = "(^|\s)--preset(?:\s+|=)structural_core_batch(\s|$)"
$trainDPattern = "(^|\s)--ablation-id(?:\s+|=)D_ablation_no_value_tree(\s|$)"

$dArtifactRelativePaths = @(
    "experiment_records\ablations\D_ablation_no_value_tree\run_record.md",
    "experiment_records\ablations\D_ablation_no_value_tree\logs\ablation_manifest.json",
    "checkpoint_store\ablations\D_ablation_no_value_tree.pt"
)

$cExtraTrainArgs = "--total-env-steps 24000 --warmup-steps 4000 --collect-steps-per-iter 16 --learner-updates-per-iter 1 --train-every-env-steps 16 --batch-size 128 --min-replay-size 8000 --replay-capacity 100000 --gamma 0.99 --n-step 3 --learning-rate 0.0001 --target-update-interval 1000 --epsilon-start 1.0 --epsilon-end 0.04 --epsilon-decay-steps 240000 --final-greedy-episodes 2 --rows 40 --cols 60 --obs-size 6 --scan-radius 10 --obstacle-ratio 0.20 --max-episode-steps 600 --coverage-stop-threshold 0.95 --fixed-train-episode-seed-base 20259323 --fixed-final-probe-seed-base 20261323 --reward-info-scale 3.1 --reward-obstacle-weight 0.2 --reward-step-penalty 0.02 --reward-terminal-bonus 20.0 --reward-revisit-penalty 0.1 --reward-turn-penalty-scale 0.05 --reward-timeout-penalty 8.0"

$cArgs = @(
    ".\experiments\baselines\run_baseline_batch.py",
    "--baseline-id", "C_baseline_local_state_ddqn",
    "--run-stage", "pilot",
    "--device", $Device,
    "--extra-train-args", $cExtraTrainArgs
)

function Quote-CommandPart {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Format-Command {
    param([string[]]$Parts)
    return (($Parts | ForEach-Object { Quote-CommandPart $_ }) -join " ")
}

function Test-DCommandLine {
    param([string]$CommandLine)

    if (-not ($CommandLine -match $formalStagePattern)) {
        return $false
    }

    $isBatchD = (
        $CommandLine -match $batchScriptPattern -and
        (
            $CommandLine -match $batchDPattern -or
            $CommandLine -match $batchStructuralPattern
        )
    )
    $isTrainD = (
        $CommandLine -match $trainScriptPattern -and
        $CommandLine -match $trainDPattern
    )

    return ($isBatchD -or $isTrainD)
}

function Get-DStructuralAblationProcesses {
    $processes = Get-CimInstance Win32_Process -ErrorAction Stop
    return @(
        $processes | Where-Object {
            $commandLine = $_.CommandLine
            $name = $_.Name
            $commandLine -and
                $name -match $pythonProcessNamePattern -and
                (Test-DCommandLine -CommandLine $commandLine)
        } | Sort-Object -Property ProcessId
    )
}

function Write-ProcessList {
    param([object[]]$Processes)

    if ($Processes.Count -eq 0) {
        Write-Host "No running D ablation process was detected."
        return
    }

    Write-Host "Detected D ablation process(es):"
    foreach ($process in $Processes) {
        Write-Host "  PID $($process.ProcessId): $($process.CommandLine)"
    }
}

function Get-DArtifactPaths {
    return @(
        $dArtifactRelativePaths | ForEach-Object {
            Join-Path $repoRoot $_
        }
    )
}

function Write-DArtifactPaths {
    param([string[]]$Paths)

    Write-Host "D artifact check paths:"
    foreach ($path in $Paths) {
        Write-Host "  $path"
    }
}

function Write-RunRecordReviewLines {
    param([string]$RunRecordPath)

    if (-not (Test-Path -LiteralPath $RunRecordPath -PathType Leaf)) {
        return
    }

    Write-Host "D run_record key lines:"
    $patterns = @(
        "checkpoint_source",
        "checkpoint_store_path",
        "checkpoint_copied",
        "checkpoint_copy_reason",
        "missing artifact list",
        "eligibility verdict"
    )
    $lines = Get-Content -LiteralPath $RunRecordPath
    foreach ($line in $lines) {
        foreach ($pattern in $patterns) {
            if ($line -match [regex]::Escape($pattern)) {
                Write-Host "  $line"
                break
            }
        }
    }
}

function Test-DArtifacts {
    param([string[]]$Paths)

    $missing = @(
        $Paths | Where-Object {
            -not (Test-Path -LiteralPath $_ -PathType Leaf)
        }
    )

    if ($missing.Count -gt 0) {
        Write-Warning "D artifact check failed. Missing path(s):"
        foreach ($path in $missing) {
            Write-Warning "  $path"
        }
        return $false
    }

    Write-Host "D artifact check passed."
    return $true
}

Write-Host "Repository root: $repoRoot"
Write-Host "Polling interval: $PollSeconds second(s)"
Write-Host "Next C command:"
Write-Host ("  " + (Format-Command (@("python") + $cArgs)))

$dArtifactPaths = @(Get-DArtifactPaths)
Write-DArtifactPaths -Paths $dArtifactPaths

$matchedProcesses = @(Get-DStructuralAblationProcesses)
Write-ProcessList -Processes $matchedProcesses

if ($DryRun) {
    Write-Host "DryRun: no waiting and no C baseline launch will be performed."
    exit 0
}

if ($matchedProcesses.Count -eq 0 -and -not $Force) {
    Write-Warning "Launching C baseline pilot without a detected D formal process may start it before D is complete."
    $confirmation = Read-Host "Type LAUNCH to start C_baseline_local_state_ddqn pilot now, or press Enter to cancel"
    if ($confirmation -cne "LAUNCH") {
        Write-Host "Cancelled; C_baseline_local_state_ddqn pilot was not launched."
        exit 0
    }
}

if ($matchedProcesses.Count -gt 0) {
    $matchedProcessIds = @($matchedProcesses | ForEach-Object { [int]$_.ProcessId })
    Write-Host "Waiting for matched D process ID(s) to exit: $($matchedProcessIds -join ', ')"

    while ($true) {
        $runningMatchedProcesses = @(
            Get-DStructuralAblationProcesses |
                Where-Object { $matchedProcessIds -contains [int]$_.ProcessId }
        )

        if ($runningMatchedProcesses.Count -eq 0) {
            break
        }

        Write-Host "Still running: $((@($runningMatchedProcesses | ForEach-Object { $_.ProcessId })) -join ', ')"
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Host "All matched D ablation process(es) exited."
}

$runRecordPath = Join-Path $repoRoot "experiment_records\ablations\D_ablation_no_value_tree\run_record.md"
Write-RunRecordReviewLines -RunRecordPath $runRecordPath

if (-not $SkipArtifactCheck) {
    $artifactsOk = Test-DArtifacts -Paths $dArtifactPaths
    if (-not $artifactsOk) {
        Write-Error -Message "D artifacts are incomplete. C baseline pilot was not launched. Use -SkipArtifactCheck only after manual verification." -ErrorAction Continue
        exit 1
    }
} else {
    Write-Warning "Skipping D artifact check because -SkipArtifactCheck was provided."
}

Write-Host "Starting C_baseline_local_state_ddqn pilot."
Push-Location $repoRoot
try {
    & python @cArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error -Message "C_baseline_local_state_ddqn pilot failed with exit code $exitCode." -ErrorAction Continue
    }
    exit $exitCode
}
finally {
    Pop-Location
}
