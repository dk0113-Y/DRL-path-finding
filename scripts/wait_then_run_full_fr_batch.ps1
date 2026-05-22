<#
.SYNOPSIS
Wait for the recommended first ablation batch, then run the full FR batch.

.DESCRIPTION
Detects currently running Python processes for:
  experiments\ablations\run_ablation_batch.py --preset recommended_first_batch --run-stage formal

If one or more matching processes are found, waits for those processes to exit, then starts:
  python experiments\ablations\run_ablation_batch.py --preset full_fr_batch --run-stage formal --device <Device> --skip-existing-records

If no matching process is found, asks for explicit confirmation before launching unless -Force is provided.

.EXAMPLE
.\scripts\wait_then_run_full_fr_batch.ps1 -DryRun

.EXAMPLE
.\scripts\wait_then_run_full_fr_batch.ps1 -Device cuda

.EXAMPLE
.\scripts\wait_then_run_full_fr_batch.ps1 -Device cuda -Force
#>

[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [ValidateRange(1, 86400)]
    [int]$PollSeconds = 60,
    [switch]$Force,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent (Resolve-Path -LiteralPath $MyInvocation.MyCommand.Path)
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDir "..")).Path

$currentScriptPattern = "run_ablation_batch\.py"
$currentPresetPattern = "(^|\s)--preset\s+recommended_first_batch(\s|$)"
$currentStagePattern = "(^|\s)--run-stage\s+formal(\s|$)"
$nextPresetPattern = "(^|\s)--preset\s+full_fr_batch(\s|$)"

$nextArgs = @(
    "experiments\ablations\run_ablation_batch.py",
    "--preset", "full_fr_batch",
    "--run-stage", "formal",
    "--device", $Device,
    "--skip-existing-records"
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

function Get-RecommendedFirstBatchProcesses {
    $processes = Get-CimInstance Win32_Process -ErrorAction Stop
    return @(
        $processes | Where-Object {
            $commandLine = $_.CommandLine
            $name = $_.Name
            $commandLine -and
                $name -match '^(python|pythonw|python3|py)(\.exe)?$' -and
                $commandLine -match $currentScriptPattern -and
                $commandLine -match $currentPresetPattern -and
                $commandLine -match $currentStagePattern -and
                $commandLine -notmatch $nextPresetPattern
        } | Sort-Object -Property ProcessId
    )
}

function Write-ProcessList {
    param([object[]]$Processes)

    if ($Processes.Count -eq 0) {
        Write-Host "No running recommended_first_batch formal process was detected."
        return
    }

    Write-Host "Detected recommended_first_batch formal process(es):"
    foreach ($process in $Processes) {
        Write-Host "  PID $($process.ProcessId): $($process.CommandLine)"
    }
}

Write-Host "Repository root: $repoRoot"
Write-Host "Polling interval: $PollSeconds second(s)"
Write-Host "Next command:"
Write-Host ("  " + (Format-Command (@("python") + $nextArgs)))

$matchedProcesses = @(Get-RecommendedFirstBatchProcesses)
Write-ProcessList -Processes $matchedProcesses

if ($DryRun) {
    Write-Host "DryRun: no waiting and no batch launch will be performed."
    exit 0
}

if ($matchedProcesses.Count -eq 0 -and -not $Force) {
    Write-Warning "Launching full_fr_batch without a detected recommended_first_batch process may start the next batch too early."
    $confirmation = Read-Host "Type LAUNCH to start full_fr_batch now, or press Enter to cancel"
    if ($confirmation -cne "LAUNCH") {
        Write-Host "Cancelled; full_fr_batch was not launched."
        exit 0
    }
}

if ($matchedProcesses.Count -gt 0) {
    $matchedProcessIds = @($matchedProcesses | ForEach-Object { [int]$_.ProcessId })
    Write-Host "Waiting for matched process ID(s) to exit: $($matchedProcessIds -join ', ')"

    while ($true) {
        $runningMatchedProcesses = @(
            Get-RecommendedFirstBatchProcesses |
                Where-Object { $matchedProcessIds -contains [int]$_.ProcessId }
        )

        if ($runningMatchedProcesses.Count -eq 0) {
            break
        }

        Write-Host "Still running: $((@($runningMatchedProcesses | ForEach-Object { $_.ProcessId })) -join ', ')"
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Host "All matched recommended_first_batch process(es) exited."
}

Write-Host "Starting full_fr_batch."
Push-Location $repoRoot
try {
    & python @nextArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error -Message "full_fr_batch failed with exit code $exitCode." -ErrorAction Continue
    }
    exit $exitCode
}
finally {
    Pop-Location
}
