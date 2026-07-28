param(
    [switch]$ValidateOnly,
    [string]$ValidationPath,
    [int]$Seed = 1,
    [int]$Step = 8,
    [int]$ScanRadius = 10,
    [string]$BlueprintPath,
    [string]$StyleContractPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [IO.Path]::GetFullPath((Join-Path $scriptDirectory "..\.."))
. (Join-Path $scriptDirectory "visio_common.ps1")
. (Join-Path $scriptDirectory "validate_fig2_environment_model.ps1")

function Write-VisioStage {
    param([Parameter(Mandatory = $true)][string]$Stage)
    Write-Host "visio_stage=$Stage"
}

function Resolve-FigureBlueprint {
    param(
        [string]$RequestedPath,
        [int]$SeedValue,
        [int]$StepValue,
        [int]$ScanRadiusValue
    )
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $resolved = [IO.Path]::GetFullPath($RequestedPath)
        if (-not (Test-Path -LiteralPath $resolved)) {
            throw "Blueprint file was not found at '$resolved'."
        }
        return $resolved
    }
    $generatedPath = Join-Path (
        [IO.Path]::GetTempPath()
    ) ("figure_demo_blueprint_seed{0}_step{1}_{2}.json" -f $SeedValue, $StepValue, $PID)
    $generatorScript = Join-Path $repoRoot "tools\export_figure_demo_blueprint.py"
    $generatorOutput = & python $generatorScript `
        --output $generatedPath `
        --seed $SeedValue `
        --step $StepValue `
        --scan-radius $ScanRadiusValue `
        --visual-ray-count 32 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Blueprint generation failed:`n$($generatorOutput -join [Environment]::NewLine)"
    }
    return $generatedPath
}

$outputDirectory = "C:\Users\Dk\Desktop\SCI\paper_picture\visio_outputs"
$requestedOutputPath = Join-Path $outputDirectory "fig2_local_grid_radar_action_space.vsdx"
$requestedPngPath = Join-Path $outputDirectory "fig2_local_grid_radar_action_space.png"
$resolvedBlueprintPath = Resolve-FigureBlueprint `
    -RequestedPath $BlueprintPath `
    -SeedValue $Seed `
    -StepValue $Step `
    -ScanRadiusValue $ScanRadius
$blueprint = Get-Content -LiteralPath $resolvedBlueprintPath -Raw | ConvertFrom-Json
$style = if ([string]::IsNullOrWhiteSpace($StyleContractPath)) {
    Import-PaperFigureStyle
}
else {
    Import-PaperFigureStyle -Path $StyleContractPath
}

if ([int]$blueprint.scan_radius -ne 10) {
    throw "Figure 2 requires scan_radius=10."
}
if ([int]$blueprint.local_shape[0] -ne 21 -or [int]$blueprint.local_shape[1] -ne 21) {
    throw "Figure 2 requires a 21x21 local_shape."
}

if ($ValidateOnly) {
    if ([string]::IsNullOrWhiteSpace($ValidationPath)) {
        throw "-ValidationPath is required with -ValidateOnly."
    }
    $validationSession = $null
    $validationDocument = $null
    try {
        $validationSession = Start-VisioSession -Visible $false
        $validationDocument = $validationSession.Application.Documents.OpenEx($ValidationPath, 194)
        $validationResult = Test-Fig2VisioDocument `
            -Document $validationDocument `
            -Blueprint $blueprint `
            -Style $style
        $validationDocument.Close()
        Release-VisioComObject -ComObject $validationDocument
        $validationDocument = $null
        $validationResult | ConvertTo-Json -Depth 8
    }
    finally {
        if ($null -ne $validationDocument) {
            try { $validationDocument.Close() } catch {}
            Release-VisioComObject -ComObject $validationDocument
        }
        Stop-VisioSession -Session $validationSession
    }
    return
}

$config = [ordered]@{
    PageWidthIn = 7.00
    PageHeightIn = 6.80
    GridRows = [int]$blueprint.local_shape[0]
    GridCols = [int]$blueprint.local_shape[1]
    CellSizeIn = 0.27
    GridLeftIn = 0.665
    GridBottomIn = 0.565
    CenterRow = [int]$blueprint.center_state[0]
    CenterCol = [int]$blueprint.center_state[1]
    RadarRayCount = [int]$blueprint.representative_rays.Count
    ActionLengthCells = 1.50
    ActionStartRadiusCells = 0.50
}
$layerNames = @(
    "Grid_Background",
    "Occupancy_Cells",
    "Grid_Lines",
    "Radar_Rays",
    "Radar_Boundary",
    "Action_Arrows",
    "Robot"
)

$session = $null
$workspace = $null
$document = $null
$page = $null
$layersByName = @{}
$radarGroup = $null
$actionGroup = $null
$robotGroup = $null
$radarBoundary = $null
$outputPair = $null
$pngResult = $null

try {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    $outputPair = Resolve-VisioOutputPair -VsdxPath $requestedOutputPath -PngPath $requestedPngPath
    $outputPath = [string]$outputPair.VsdxPath
    $pngPath = [string]$outputPair.PngPath
    $session = Start-VisioSession -Visible $false
    Write-VisioStage -Stage "session_started"
    $workspace = New-VisioDocumentPage `
        -Session $session `
        -PageWidthIn ([double]$config.PageWidthIn) `
        -PageHeightIn ([double]$config.PageHeightIn) `
        -PageName "Fig2_Local_Grid_Radar_Actions"
    $document = $workspace.Document
    $page = $workspace.Page
    foreach ($layerName in $layerNames) {
        $layersByName[$layerName] = New-VisioLayer -Page $page -Name $layerName
    }
    Write-VisioStage -Stage "layers_created"

    $cellSize = [double]$config.CellSizeIn
    $gridLeft = [double]$config.GridLeftIn
    $gridBottom = [double]$config.GridBottomIn
    $gridRight = $gridLeft + ([int]$config.GridCols * $cellSize)
    $gridTop = $gridBottom + ([int]$config.GridRows * $cellSize)
    $centerX = $gridLeft + (([int]$config.CenterCol + 0.5) * $cellSize)
    $centerY = $gridBottom + (([int]$config.GridRows - [int]$config.CenterRow - 0.5) * $cellSize)

    $background = New-GridCell `
        -Page $page `
        -Layer $layersByName["Grid_Background"] `
        -Name "Local_Grid_Background" `
        -Left $gridLeft `
        -Bottom $gridBottom `
        -Right $gridRight `
        -Top $gridTop `
        -FillColor ([string]$style.OccupancyPalette.unknown)
    Release-VisioComObject -ComObject $background

    for ($row = 0; $row -lt [int]$config.GridRows; $row++) {
        for ($col = 0; $col -lt [int]$config.GridCols; $col++) {
            $state = [int]$blueprint.local_snap_t[$row][$col]
            $fillColor = switch ($state) {
                -1 { [string]$style.OccupancyPalette.unknown }
                0 { [string]$style.OccupancyPalette.free }
                1 { [string]$style.OccupancyPalette.obstacle }
                default { throw "Unexpected local_snap value $state at ($row,$col)." }
            }
            $left = $gridLeft + ($col * $cellSize)
            $bottom = $gridBottom + (([int]$config.GridRows - 1 - $row) * $cellSize)
            $cell = New-GridCell `
                -Page $page `
                -Layer $layersByName["Occupancy_Cells"] `
                -Name ("Cell_r{0:D2}_c{1:D2}" -f $row, $col) `
                -Left $left `
                -Bottom $bottom `
                -Right ($left + $cellSize) `
                -Top ($bottom + $cellSize) `
                -FillColor $fillColor
            Release-VisioComObject -ComObject $cell
        }
    }
    Write-VisioStage -Stage "occupancy_cells_created"

    for ($index = 0; $index -le [int]$config.GridCols; $index++) {
        $x = $gridLeft + ($index * $cellSize)
        $line = New-GridLine `
            -Page $page `
            -Layer $layersByName["Grid_Lines"] `
            -Name ("GridLine_V_{0:D2}" -f $index) `
            -BeginX $x -BeginY $gridBottom -EndX $x -EndY $gridTop `
            -LineColor ([string]$style.RadarPalette.grid_line) -LineWeightPt 0.55
        Release-VisioComObject -ComObject $line
    }
    for ($index = 0; $index -le [int]$config.GridRows; $index++) {
        $y = $gridBottom + ($index * $cellSize)
        $line = New-GridLine `
            -Page $page `
            -Layer $layersByName["Grid_Lines"] `
            -Name ("GridLine_H_{0:D2}" -f $index) `
            -BeginX $gridLeft -BeginY $y -EndX $gridRight -EndY $y `
            -LineColor ([string]$style.RadarPalette.grid_line) -LineWeightPt 0.55
        Release-VisioComObject -ComObject $line
    }
    Write-VisioStage -Stage "grid_lines_created"

    $radarRays = @()
    for ($rayIndex = 0; $rayIndex -lt [int]$config.RadarRayCount; $rayIndex++) {
        $endRow = [int]$blueprint.representative_rays[$rayIndex].end_local_rc[0]
        $endCol = [int]$blueprint.representative_rays[$rayIndex].end_local_rc[1]
        $endX = $centerX + (($endCol - [int]$config.CenterCol) * $cellSize)
        $endY = $centerY - (($endRow - [int]$config.CenterRow) * $cellSize)
        $radarRays += New-LineShape `
            -Page $page `
            -Layer $layersByName["Radar_Rays"] `
            -Name ("Radar_Ray_{0:D2}" -f ($rayIndex + 1)) `
            -BeginX $centerX -BeginY $centerY -EndX $endX -EndY $endY `
            -LineColor ([string]$style.RadarPalette.ray) `
            -LineWeightPt 0.72 `
            -TransparencyPercent 54.0
    }
    $radarGroup = New-VisioShapeGroup `
        -Page $page -Shapes $radarRays -Name "Radar_Rays_Group" -Layer $layersByName["Radar_Rays"]
    foreach ($shape in $radarRays) { Release-VisioComObject -ComObject $shape }
    Write-VisioStage -Stage "radar_rays_created"

    $radarRadiusIn = [int]$blueprint.scan_radius * $cellSize
    $radarBoundary = $page.DrawOval(
        $centerX - $radarRadiusIn,
        $centerY - $radarRadiusIn,
        $centerX + $radarRadiusIn,
        $centerY + $radarRadiusIn
    )
    Set-VisioShapeName -Shape $radarBoundary -Name "Radar_Range_Boundary"
    Set-VisioCellFormula -Shape $radarBoundary -CellName "FillPattern" -Formula "0"
    Set-VisioLineStyle `
        -Shape $radarBoundary `
        -LineColor ([string]$style.RadarPalette.nominal_boundary) `
        -LineWeightPt 1.05 `
        -TransparencyPercent 35.0
    Set-VisioCellFormula -Shape $radarBoundary -CellName "LinePattern" -Formula "2"
    Add-ShapeToVisioLayer -Layer $layersByName["Radar_Boundary"] -Shape $radarBoundary

    $directions = @(
        [pscustomobject]@{ Name = "Action_N"; X = 0.0; Y = 1.0 },
        [pscustomobject]@{ Name = "Action_NE"; X = 1.0; Y = 1.0 },
        [pscustomobject]@{ Name = "Action_E"; X = 1.0; Y = 0.0 },
        [pscustomobject]@{ Name = "Action_SE"; X = 1.0; Y = -1.0 },
        [pscustomobject]@{ Name = "Action_S"; X = 0.0; Y = -1.0 },
        [pscustomobject]@{ Name = "Action_SW"; X = -1.0; Y = -1.0 },
        [pscustomobject]@{ Name = "Action_W"; X = -1.0; Y = 0.0 },
        [pscustomobject]@{ Name = "Action_NW"; X = -1.0; Y = 1.0 }
    )
    $actionArrows = @()
    $actionStartRadiusIn = [double]$config.ActionStartRadiusCells * $cellSize
    $actionLengthIn = [double]$config.ActionLengthCells * $cellSize
    foreach ($direction in $directions) {
        $norm = [Math]::Sqrt(
            ([double]$direction.X * [double]$direction.X) +
            ([double]$direction.Y * [double]$direction.Y)
        )
        $unitX = [double]$direction.X / $norm
        $unitY = [double]$direction.Y / $norm
        $beginX = $centerX + ($unitX * $actionStartRadiusIn)
        $beginY = $centerY + ($unitY * $actionStartRadiusIn)
        $actionArrows += New-ArrowShape `
            -Page $page `
            -Layer $layersByName["Action_Arrows"] `
            -Name ([string]$direction.Name) `
            -BeginX $beginX -BeginY $beginY `
            -EndX ($beginX + ($unitX * $actionLengthIn)) `
            -EndY ($beginY + ($unitY * $actionLengthIn)) `
            -LineColor ([string]$style.RadarPalette.action) `
            -LineWeightPt 2.25 -EndArrow 13 -EndArrowSize 3
    }
    $actionGroup = New-VisioShapeGroup `
        -Page $page -Shapes $actionArrows -Name "Action_Arrows_Group" -Layer $layersByName["Action_Arrows"]
    foreach ($shape in $actionArrows) { Release-VisioComObject -ComObject $shape }

    $robotParts = Add-PaperFigureRobotParts `
        -Page $page `
        -Layer $layersByName["Robot"] `
        -NamePrefix "Robot" `
        -CenterX $centerX `
        -CenterY $centerY `
        -CellSizeIn $cellSize `
        -Style $style `
        -HeadingAction ([int]$blueprint.selected_action)
    $robotGroup = New-VisioShapeGroup `
        -Page $page -Shapes $robotParts -Name "Robot_Group" -Layer $layersByName["Robot"]
    foreach ($shape in $robotParts) { Release-VisioComObject -ComObject $shape }
    $robotGroup.BringToFront()
    Write-VisioStage -Stage "native_scene_created"

    $null = $document.SaveAs($outputPath)
    $pngResult = Export-VisioPng -Page $page -Path $pngPath -Dpi 300
    Write-VisioStage -Stage "document_saved_and_png_exported"
    $document.Close()

    Release-VisioComObject -ComObject $radarGroup
    Release-VisioComObject -ComObject $radarBoundary
    Release-VisioComObject -ComObject $actionGroup
    Release-VisioComObject -ComObject $robotGroup
    $radarGroup = $null
    $radarBoundary = $null
    $actionGroup = $null
    $robotGroup = $null
    foreach ($layer in $layersByName.Values) { Release-VisioComObject -ComObject $layer }
    $layersByName = @{}
    Release-VisioComObject -ComObject $page
    $page = $null
    $workspace.Page = $null
    $workspace.Document = $null
    Release-VisioComObject -ComObject $document
    $document = $null
    Stop-VisioSession -Session $session
    $session = $null

    $validationOutput = & powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $MyInvocation.MyCommand.Path `
        -ValidateOnly `
        -ValidationPath $outputPath `
        -BlueprintPath $resolvedBlueprintPath `
        -StyleContractPath $style.ContractPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Visio reopen validation failed with code $LASTEXITCODE."
    }
    $validationJson = $validationOutput -join [Environment]::NewLine
    $validation = $validationJson | ConvertFrom-Json
    $summaryPath = Join-Path (
        [IO.Path]::GetTempPath()
    ) ("fig2_visio_validation_{0}_{1}.json" -f $PID, (Get-Date -Format "yyyyMMdd_HHmmss"))
    $validationJson | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    $outputFile = Get-Item -LiteralPath $outputPath
    [ordered]@{
        OutputPath = $outputFile.FullName
        PngPath = $pngResult.Path
        FileSizeBytes = [long]$outputFile.Length
        PngFileSizeBytes = [long]$pngResult.FileSizeBytes
        PngPixelWidth = [int]$pngResult.PixelWidth
        PngPixelHeight = [int]$pngResult.PixelHeight
        PngDpi = [int]$pngResult.Dpi
        BlueprintPath = $resolvedBlueprintPath
        StyleContractPath = [string]$style.ContractPath
        StyleContractVersion = [string]$style.Version
        ReopenValidation = "passed"
        ValidationSummaryPath = $summaryPath
        Validation = $validation
    } | ConvertTo-Json -Depth 10
}
catch {
    throw "Failed to build or validate Figure 2 Visio document: $($_.Exception.Message)`n$($_.ScriptStackTrace)"
}
finally {
    if ($null -ne $document) {
        try { $document.Close() } catch {}
    }
    Release-VisioComObject -ComObject $radarGroup
    Release-VisioComObject -ComObject $radarBoundary
    Release-VisioComObject -ComObject $actionGroup
    Release-VisioComObject -ComObject $robotGroup
    foreach ($layer in $layersByName.Values) { Release-VisioComObject -ComObject $layer }
    Release-VisioComObject -ComObject $page
    if ($null -ne $workspace) {
        $workspace.Page = $null
        $workspace.Document = $null
    }
    Release-VisioComObject -ComObject $document
    Stop-VisioSession -Session $session
}
