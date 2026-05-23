<#
.SYNOPSIS
Wait for C local-state DDQN formal training, then run the E structural ablation formal training.

.DESCRIPTION
Detects currently running Python processes for C group formal baseline work:
  python experiments\baselines\run_baseline_batch.py --baseline-id C_baseline_local_state_ddqn --run-stage formal
  python experiments\baselines\run_local_state_ddqn_train.py --baseline-id C_baseline_local_state_ddqn --run-stage formal

If one or more matching processes are found, waits for those processes to exit.
After C exits, checks the key curated C artifacts and then starts:
  python .\experiments\ablations\run_ablation_batch.py --ablation-ids E --run-stage formal --device <Device>

If no matching C formal process is found, asks for explicit confirmation before
launching unless -Force is provided. Use -SkipArtifactCheck only after manually
confirming that skipping the C artifact gate is intentional.

C formal completion here means the training-side record/checkpoint archive is
available for handoff. It does not mean the C run is final paper Results
evidence; train-side-only formal runs may still report
unable_to_judge_for_final_results and may lack final_probe artifacts.

E formal retrains E_ablation_no_semantic_dual_state_split through the ablation
batch runner so curated logs and checkpoints are archived consistently. E keeps
value-tree information and removes the semantic dual-state split decision
structure; it is not the D no-value-tree ablation.

.EXAMPLE
.\scripts\wait_then_run_e_ablation_after_c_baseline.ps1 -DryRun

.EXAMPLE
.\scripts\wait_then_run_e_ablation_after_c_baseline.ps1 -Device cuda

.EXAMPLE
.\scripts\wait_then_run_e_ablation_after_c_baseline.ps1 -Device cuda -Force

.EXAMPLE
.\scripts\wait_then_run_e_ablation_after_c_baseline.ps1 -Device cuda -Force -SkipArtifactCheck
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

function Find-RepositoryRoot {
    param([string]$StartDirectory)

    $current = (Resolve-Path -LiteralPath $StartDirectory).Path
    while ($true) {
        if (Test-Path -LiteralPath (Join-Path $current ".git")) {
            return $current
        }

        $parent = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $current) {
            throw "Could not locate repository root from $StartDirectory"
        }
        $current = $parent
    }
}

$repoRoot = Find-RepositoryRoot -StartDirectory $scriptDir

$pythonProcessNamePattern = "^(python|pythonw|python3|py)(\.exe)?$"
$formalStagePattern = "(^|\s)--run-stage(?:\s+|=)formal(\s|$)"
$baselineIdPattern = "(^|\s)--baseline-id(?:\s+|=)C_baseline_local_state_ddqn(\s|$)"
$batchScriptPattern = "experiments[\\/]+baselines[\\/]+run_baseline_batch\.py|run_baseline_batch\.py"
$trainScriptPattern = "experiments[\\/]+baselines[\\/]+run_local_state_ddqn_train\.py|run_local_state_ddqn_train\.py"

$cArtifactRelativePaths = @(
    "experiment_records\baselines\C_baseline_local_state_ddqn\run_record.md",
    "experiment_records\baselines\C_baseline_local_state_ddqn\logs\baseline_manifest.json",
    "checkpoint_store\baselines\C_baseline_local_state_ddqn.pt"
)

$eArgs = @(
    ".\experiments\ablations\run_ablation_batch.py",
    "--ablation-ids", "E",
    "--run-stage", "formal",
    "--device", $Device
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

function Test-CFormalCommandLine {
    param([string]$CommandLine)

    if (-not ($CommandLine -match $formalStagePattern)) {
        return $false
    }
    if (-not ($CommandLine -match $baselineIdPattern)) {
        return $false
    }

    return (
        $CommandLine -match $batchScriptPattern -or
        $CommandLine -match $trainScriptPattern
    )
}

function Get-CBaselineFormalProcesses {
    $processes = Get-CimInstance Win32_Process -ErrorAction Stop
    return @(
        $processes | Where-Object {
            $commandLine = $_.CommandLine
            $name = $_.Name
            $commandLine -and
                $name -match $pythonProcessNamePattern -and
                (Test-CFormalCommandLine -CommandLine $commandLine)
        } | Sort-Object -Property ProcessId
    )
}

function Write-ProcessList {
    param([object[]]$Processes)

    if ($Processes.Count -eq 0) {
        Write-Host "No running C baseline formal process was detected."
        return
    }

    Write-Host "Detected C baseline formal process(es):"
    foreach ($process in $Processes) {
        Write-Host "  PID $($process.ProcessId): $($process.CommandLine)"
    }
}

function Get-CArtifactPaths {
    return @(
        $cArtifactRelativePaths | ForEach-Object {
            Join-Path $repoRoot $_
        }
    )
}

function Write-CArtifactPaths {
    param([string[]]$Paths)

    Write-Host "C artifact check paths:"
    foreach ($path in $Paths) {
        Write-Host "  $path"
    }
}

function Write-RunRecordReviewLines {
    param([string]$RunRecordPath)

    if (-not (Test-Path -LiteralPath $RunRecordPath -PathType Leaf)) {
        return
    }

    Write-Host "C run_record key lines:"
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

function Test-CArtifacts {
    param([string[]]$Paths)

    $missing = @(
        $Paths | Where-Object {
            -not (Test-Path -LiteralPath $_ -PathType Leaf)
        }
    )

    if ($missing.Count -gt 0) {
        Write-Warning "C artifact check failed. Missing path(s):"
        foreach ($path in $missing) {
            Write-Warning "  $path"
        }
        return $false
    }

    Write-Host "C artifact check passed."
    return $true
}

$eCommandParts = @("python") + $eArgs
$eCommandText = Format-Command $eCommandParts
if ($eCommandText -match "--trajectory-history-steps") {
    throw "Internal safety check failed: E command must not contain --trajectory-history-steps."
}

Write-Host "Repository root: $repoRoot"
Write-Host "Polling interval: $PollSeconds second(s)"
Write-Host "Next E command:"
Write-Host ("  " + $eCommandText)

$cArtifactPaths = @(Get-CArtifactPaths)
Write-CArtifactPaths -Paths $cArtifactPaths

$matchedProcesses = @(Get-CBaselineFormalProcesses)
Write-ProcessList -Processes $matchedProcesses

if ($DryRun) {
    Write-Host "DryRun: no waiting and no E ablation launch will be performed."
    exit 0
}

if ($matchedProcesses.Count -eq 0 -and -not $Force) {
    Write-Warning "Launching E formal without a detected C formal process may start it before C is complete."
    $confirmation = Read-Host "Type LAUNCH to start E_ablation_no_semantic_dual_state_split formal now, or press Enter to cancel"
    if ($confirmation -cne "LAUNCH") {
        Write-Host "Cancelled; E_ablation_no_semantic_dual_state_split formal was not launched."
        exit 0
    }
}

if ($matchedProcesses.Count -gt 0) {
    $matchedProcessIds = @($matchedProcesses | ForEach-Object { [int]$_.ProcessId })
    Write-Host "Waiting for matched C process ID(s) to exit: $($matchedProcessIds -join ', ')"

    while ($true) {
        $runningMatchedProcesses = @(
            Get-CBaselineFormalProcesses |
                Where-Object { $matchedProcessIds -contains [int]$_.ProcessId }
        )

        if ($runningMatchedProcesses.Count -eq 0) {
            break
        }

        Write-Host "Still running: $((@($runningMatchedProcesses | ForEach-Object { $_.ProcessId })) -join ', ')"
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Host "All matched C baseline formal process(es) exited."
}

$runRecordPath = Join-Path $repoRoot "experiment_records\baselines\C_baseline_local_state_ddqn\run_record.md"
Write-RunRecordReviewLines -RunRecordPath $runRecordPath

if (-not $SkipArtifactCheck) {
    $artifactsOk = Test-CArtifacts -Paths $cArtifactPaths
    if (-not $artifactsOk) {
        Write-Error -Message "C artifacts are incomplete. E ablation formal was not launched. Use -SkipArtifactCheck only after manual verification." -ErrorAction Continue
        exit 1
    }
} else {
    Write-Warning "Skipping C artifact check because -SkipArtifactCheck was provided."
}

Write-Host "Starting E_ablation_no_semantic_dual_state_split formal."
Push-Location $repoRoot
try {
    & python @eArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error -Message "E_ablation_no_semantic_dual_state_split formal failed with exit code $exitCode." -ErrorAction Continue
    }
    exit $exitCode
}
finally {
    Pop-Location
}
