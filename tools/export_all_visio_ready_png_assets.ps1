[CmdletBinding()]
param(
    [string]$OutputRoot = "C:\Users\Dk\Desktop\SCI\论文0\New\绘图\fig_fix_1",
    [ValidateRange(1, 10000)]
    [int]$Dpi = 600,
    [switch]$IncludeSvg,
    [string]$PaperRepo = "",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$exportScript = Join-Path $PSScriptRoot "export_all_visio_ready_png_assets.py"
$exportArgs = @($exportScript, "--output-root", $OutputRoot, "--dpi", [string]$Dpi)
if ($IncludeSvg) {
    $exportArgs += "--include-svg"
}
if ($PaperRepo) {
    $exportArgs += @("--paper-repo", $PaperRepo)
}

& $PythonExecutable @exportArgs
if ($LASTEXITCODE -ne 0) {
    throw "Visio-ready PNG export failed with exit code $LASTEXITCODE."
}
