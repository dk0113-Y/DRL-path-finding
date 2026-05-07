[CmdletBinding()]
param(
    [string]$PythonExecutable = "python",
    [string]$RunNameA = "stable_same_seed_minreplay8000_A",
    [string]$RunNameB = "stable_same_seed_minreplay8000_B",
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
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$stableLauncher = Join-Path $scriptDir "launch_formal_train_stable.ps1"
$contractChecker = Join-Path $scriptDir "check_reproducibility_contract.py"
$outputsDir = Join-Path $repoRoot "outputs"
$powerShellExecutable = (Get-Command powershell -ErrorAction Stop).Source
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path -LiteralPath $stableLauncher -PathType Leaf)) {
    throw "Stable single-run launcher not found: $stableLauncher"
}
if (-not (Test-Path -LiteralPath $contractChecker -PathType Leaf)) {
    throw "Reproducibility contract checker not found: $contractChecker"
}

function Quote-CommandPart {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Get-SharedLauncherArgs {
    param([string]$RunName)
    return @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $stableLauncher,
        "-PythonExecutable", $PythonExecutable,
        "-RunName", $RunName,
        "-Device", $Device,
        "-Seed", ([string]$Seed),
        "-TotalEnvSteps", ([string]$TotalEnvSteps),
        "-EpsilonDecaySteps", ([string]$EpsilonDecaySteps),
        "-EpsilonEnd", ($EpsilonEnd.ToString($invariantCulture)),
        "-MinReplaySize", ([string]$MinReplaySize),
        "-FinalGreedyEpisodes", ([string]$FinalGreedyEpisodes),
        "-FixedTrainEpisodeSeedBase", ([string]$FixedTrainEpisodeSeedBase),
        "-FixedFinalProbeSeedBase", ([string]$FixedFinalProbeSeedBase)
    )
}

function Write-SharedParameterSummary {
    Write-Host "Exact shared parameters:"
    Write-Host "  PythonExecutable: $PythonExecutable"
    Write-Host "  Device: $Device"
    Write-Host "  Seed: $Seed"
    Write-Host "  TotalEnvSteps: $TotalEnvSteps"
    Write-Host "  EpsilonDecaySteps: $EpsilonDecaySteps"
    Write-Host ("  EpsilonEnd: " + $EpsilonEnd.ToString($invariantCulture))
    Write-Host "  MinReplaySize: $MinReplaySize"
    Write-Host "  FinalGreedyEpisodes: $FinalGreedyEpisodes"
    Write-Host "  FixedTrainEpisodeSeedBase: $FixedTrainEpisodeSeedBase"
    Write-Host "  FixedFinalProbeSeedBase: $FixedFinalProbeSeedBase"
    Write-Host "  Stable launcher: $stableLauncher"
    Write-Host "  Contract checker: $contractChecker"
}

function Invoke-StableLauncher {
    param(
        [string]$RunLabel,
        [string]$RunName,
        [switch]$DryRunMode
    )

    $launcherArgs = @(Get-SharedLauncherArgs -RunName $RunName)
    if ($DryRunMode) {
        $launcherArgs += "-DryRun"
    }

    Write-Host "$RunLabel name: $RunName"
    Write-Host "Single-run launcher invocation:"
    Write-Host ("  " + ((@($powerShellExecutable) + $launcherArgs | ForEach-Object { Quote-CommandPart $_ }) -join " "))

    & $powerShellExecutable @launcherArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$RunLabel stable launcher failed with exit code $exitCode."
    }
}

function Get-NewestRunDirectory {
    param(
        [string]$RunLabel,
        [string]$RunName,
        [datetime]$LaunchStartedAt
    )

    if (-not (Test-Path -LiteralPath $outputsDir -PathType Container)) {
        throw "Outputs directory not found after ${RunLabel}: $outputsDir"
    }

    $prefix = "$RunName`_"
    $matches = @(
        Get-ChildItem -LiteralPath $outputsDir -Directory -ErrorAction Stop |
            Where-Object { $_.Name.StartsWith($prefix, [System.StringComparison]::Ordinal) }
    )
    if ($matches.Count -eq 0) {
        throw "No output directory found for ${RunLabel}; expected name prefix '$prefix' under $outputsDir."
    }

    $recentMatches = @(
        $matches |
            Where-Object { $_.CreationTime -ge $LaunchStartedAt -or $_.LastWriteTime -ge $LaunchStartedAt }
    )
    if ($recentMatches.Count -eq 0) {
        throw "No newly created or modified output directory found for ${RunLabel}; expected prefix '$prefix' after $($LaunchStartedAt.ToString('o'))."
    }

    $selected = $recentMatches |
        Sort-Object -Property LastWriteTime, CreationTime -Descending |
        Select-Object -First 1
    if ($recentMatches.Count -gt 1) {
        Write-Host "Multiple matching output directories found for ${RunLabel}; selected newest: $($selected.FullName)"
    } else {
        Write-Host "Selected ${RunLabel} output directory: $($selected.FullName)"
    }
    return $selected.FullName
}

function Invoke-ContractCheck {
    param(
        [string]$RunLabel,
        [string]$RunDirectory
    )

    $contractPath = Join-Path $RunDirectory "logs\reproducibility_contract.json"
    if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf)) {
        throw "${RunLabel} reproducibility contract not found: $contractPath"
    }

    Write-Host "Validating ${RunLabel} contract: $contractPath"
    & $PythonExecutable $contractChecker $contractPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "${RunLabel} reproducibility contract check failed with exit code $exitCode."
    }
    return $contractPath
}

try {
    Write-Host "Repository root: $repoRoot"
    Write-Host "Stable same-seed A/B launcher is sequential: Run B starts only after Run A and its contract check pass."
    Write-SharedParameterSummary
    Write-Host "Run A and Run B differ only by run name: '$RunNameA' vs '$RunNameB'."

    if ($DryRun) {
        Write-Host "DryRun: calling the single-run launcher's DryRun for Run A."
        Invoke-StableLauncher -RunLabel "Run A" -RunName $RunNameA -DryRunMode
        Write-Host "DryRun: Run A contract check would be run after Run A completes."
        Write-Host "DryRun: calling the single-run launcher's DryRun for Run B."
        Invoke-StableLauncher -RunLabel "Run B" -RunName $RunNameB -DryRunMode
        Write-Host "DryRun: Run B contract check would be run after Run B completes."
        Write-Host "DryRun: A and B differ only by run name."
        Write-Host "DryRun: no Python launched by this A/B wrapper."
        exit 0
    }

    Write-Host "Starting Run A."
    $runAStartedAt = Get-Date
    Invoke-StableLauncher -RunLabel "Run A" -RunName $RunNameA
    $runADirectory = Get-NewestRunDirectory -RunLabel "Run A" -RunName $RunNameA -LaunchStartedAt $runAStartedAt
    $runAContract = Invoke-ContractCheck -RunLabel "Run A" -RunDirectory $runADirectory

    Write-Host "Run A completed and contract check passed. Starting Run B."
    $runBStartedAt = Get-Date
    Invoke-StableLauncher -RunLabel "Run B" -RunName $RunNameB
    $runBDirectory = Get-NewestRunDirectory -RunLabel "Run B" -RunName $RunNameB -LaunchStartedAt $runBStartedAt
    $runBContract = Invoke-ContractCheck -RunLabel "Run B" -RunDirectory $runBDirectory

    Write-Host "Stable same-seed A/B launch completed successfully."
    Write-Host "Run A directory: $runADirectory"
    Write-Host "Run B directory: $runBDirectory"
    Write-Host "Run A contract: $runAContract"
    Write-Host "Run B contract: $runBContract"
    Write-SharedParameterSummary
    Write-Host "Only run names differed: '$RunNameA' vs '$RunNameB'."
    exit 0
}
catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
