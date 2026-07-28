param(
    [switch]$ValidateOnly,
    [string]$ValidationPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDirectory "visio_common.ps1")
. (Join-Path $scriptDirectory "validate_fig2_environment_model.ps1")

function Write-VisioStage {
    param([Parameter(Mandatory = $true)][string]$Stage)
    Write-Host "visio_stage=$Stage"
}

$outputDirectory = "C:\Users\Dk\Desktop\SCI\paper_picture\visio_outputs"
$requestedOutputPath = Join-Path $outputDirectory "fig2_local_grid_radar_action_space.vsdx"
$requestedPngPath = Join-Path $outputDirectory "fig2_local_grid_radar_action_space.png"

if ($ValidateOnly) {
    if ([string]::IsNullOrWhiteSpace($ValidationPath)) {
        throw "-ValidationPath is required with -ValidateOnly."
    }
    $validationSession = $null
    $validationDocument = $null
    try {
        $validationSession = Start-VisioSession -Visible $false
        # visOpenRO (2) + visOpenHidden (64) + visOpenMacrosDisabled (128).
        $validationDocument = $validationSession.Application.Documents.OpenEx($ValidationPath, 194)
        $validationResult = Test-Fig2VisioDocument -Document $validationDocument
        $validationDocument.Close()
        Release-VisioComObject -ComObject $validationDocument
        $validationDocument = $null
        $validationResult | ConvertTo-Json -Depth 8
    }
    finally {
        if ($null -ne $validationDocument) {
            try {
                $validationDocument.Close()
            }
            catch {
            }
            Release-VisioComObject -ComObject $validationDocument
        }
        Stop-VisioSession -Session $validationSession
    }
    return
}

$config = [ordered]@{
    PageWidthIn = 7.40
    PageHeightIn = 6.80
    GridRows = 15
    GridCols = 15
    CellSizeIn = 0.40
    GridLeftIn = 0.70
    GridBottomIn = 0.40
    CenterRow = 7
    CenterCol = 7
    RadarRadiusCells = 5.15
    RadarFillRadiusCells = 5.15
    RadarRayCount = 32
    ActionLengthCells = 1.50
    ActionStartRadiusCells = 0.45
    UnknownColor = "#B4BCC4"
    FreeColor = "#F8F8F6"
    ObstacleColor = "#1E1E1E"
    GridLineColor = "#C8D2DA"
    RadarColor = "#5185C0"
    ActionColor = "#C96144"
    RobotBodyColor = "#55966B"
    RobotBodyEdgeColor = "#2F5940"
    RobotWheelColor = "#30363B"
    RobotRadarColor = "#E99D4E"
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
$fixedObstacleKeys = @(
    "2,7",
    "3,4",
    "3,10",
    "4,3",
    "4,11",
    "7,12",
    "9,11",
    "10,3",
    "11,9",
    "12,6"
)
$obstacleLookup = @{}
foreach ($key in $fixedObstacleKeys) {
    $obstacleLookup[$key] = $true
}

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
    $outputPair = Resolve-VisioOutputPair `
        -VsdxPath $requestedOutputPath `
        -PngPath $requestedPngPath
    $outputPath = [string]$outputPair.VsdxPath
    $pngPath = [string]$outputPair.PngPath
    if ($outputPair.UsedTimestampFallback) {
        Write-Host "visio_output_locked=$([string]::Join(';', $outputPair.LockedPaths))"
        Write-Host "visio_timestamp_fallback=$outputPath"
    }

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
        -FillColor ([string]$config.UnknownColor)
    Release-VisioComObject -ComObject $background

    for ($row = 0; $row -lt [int]$config.GridRows; $row++) {
        for ($col = 0; $col -lt [int]$config.GridCols; $col++) {
            $left = $gridLeft + ($col * $cellSize)
            $bottom = $gridBottom + (([int]$config.GridRows - 1 - $row) * $cellSize)
            $dr = $row - [int]$config.CenterRow
            $dc = $col - [int]$config.CenterCol
            $distanceCells = [Math]::Sqrt(($dr * $dr) + ($dc * $dc))
            $fillColor = if ($distanceCells -le [double]$config.RadarFillRadiusCells) {
                [string]$config.FreeColor
            }
            else {
                [string]$config.UnknownColor
            }
            if ($obstacleLookup.ContainsKey("$row,$col")) {
                $fillColor = [string]$config.ObstacleColor
            }

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
            -BeginX $x `
            -BeginY $gridBottom `
            -EndX $x `
            -EndY $gridTop `
            -LineColor ([string]$config.GridLineColor) `
            -LineWeightPt 0.55
        Release-VisioComObject -ComObject $line
    }
    for ($index = 0; $index -le [int]$config.GridRows; $index++) {
        $y = $gridBottom + ($index * $cellSize)
        $line = New-GridLine `
            -Page $page `
            -Layer $layersByName["Grid_Lines"] `
            -Name ("GridLine_H_{0:D2}" -f $index) `
            -BeginX $gridLeft `
            -BeginY $y `
            -EndX $gridRight `
            -EndY $y `
            -LineColor ([string]$config.GridLineColor) `
            -LineWeightPt 0.55
        Release-VisioComObject -ComObject $line
    }
    Write-VisioStage -Stage "grid_lines_created"

    # Representative equal-length rays for schematic visualization only.
    $radarRays = @()
    $radarRadiusIn = [double]$config.RadarRadiusCells * $cellSize
    for ($rayIndex = 0; $rayIndex -lt [int]$config.RadarRayCount; $rayIndex++) {
        $angle = (2.0 * [Math]::PI * $rayIndex) / [int]$config.RadarRayCount
        $endX = $centerX + ([Math]::Sin($angle) * $radarRadiusIn)
        $endY = $centerY + ([Math]::Cos($angle) * $radarRadiusIn)
        $radarRays += New-LineShape `
            -Page $page `
            -Layer $layersByName["Radar_Rays"] `
            -Name ("Radar_Ray_{0:D2}" -f ($rayIndex + 1)) `
            -BeginX $centerX `
            -BeginY $centerY `
            -EndX $endX `
            -EndY $endY `
            -LineColor ([string]$config.RadarColor) `
            -LineWeightPt 0.72 `
            -TransparencyPercent 54.0
    }
    $radarGroup = New-VisioShapeGroup `
        -Page $page `
        -Shapes $radarRays `
        -Name "Radar_Rays_Group" `
        -Layer $layersByName["Radar_Rays"]
    foreach ($shape in $radarRays) {
        Release-VisioComObject -ComObject $shape
    }
    $radarRays = @()
    Write-VisioStage -Stage "radar_rays_created"

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
        -LineColor ([string]$config.RadarColor) `
        -LineWeightPt 1.05 `
        -TransparencyPercent 16.0
    Add-ShapeToVisioLayer -Layer $layersByName["Radar_Boundary"] -Shape $radarBoundary
    Write-VisioStage -Stage "radar_boundary_created"

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
            -BeginX $beginX `
            -BeginY $beginY `
            -EndX ($beginX + ($unitX * $actionLengthIn)) `
            -EndY ($beginY + ($unitY * $actionLengthIn)) `
            -LineColor ([string]$config.ActionColor) `
            -LineWeightPt 2.25 `
            -EndArrow 13 `
            -EndArrowSize 3
    }
    $actionGroup = New-VisioShapeGroup `
        -Page $page `
        -Shapes $actionArrows `
        -Name "Action_Arrows_Group" `
        -Layer $layersByName["Action_Arrows"]
    foreach ($shape in $actionArrows) {
        Release-VisioComObject -ComObject $shape
    }
    $actionArrows = @()
    Write-VisioStage -Stage "action_arrows_created"

    $robotParts = @()
    $wheelWidth = 0.045
    $wheelHeight = 0.070
    $wheelOffsetX = 0.1375
    $wheelOffsetY = 0.095
    $wheelIndex = 0
    foreach ($offsetY in @(-$wheelOffsetY, $wheelOffsetY)) {
        foreach ($offsetX in @(-$wheelOffsetX, $wheelOffsetX)) {
            $wheelIndex++
            $wheel = $page.DrawRectangle(
                $centerX + $offsetX - ($wheelWidth / 2.0),
                $centerY + $offsetY - ($wheelHeight / 2.0),
                $centerX + $offsetX + ($wheelWidth / 2.0),
                $centerY + $offsetY + ($wheelHeight / 2.0)
            )
            Set-VisioShapeName -Shape $wheel -Name ("Robot_Wheel_{0:D2}" -f $wheelIndex)
            Set-VisioNodeStyle `
                -Shape $wheel `
                -FillColor ([string]$config.RobotWheelColor) `
                -LineColor ([string]$config.RobotWheelColor) `
                -LineWeightPt 0.55 `
                -RoundingIn 0.018
            Add-ShapeToVisioLayer -Layer $layersByName["Robot"] -Shape $wheel
            $robotParts += $wheel
        }
    }

    $bodyWidth = 0.245
    $bodyHeight = 0.270
    $body = $page.DrawRectangle(
        $centerX - ($bodyWidth / 2.0),
        $centerY - ($bodyHeight / 2.0),
        $centerX + ($bodyWidth / 2.0),
        $centerY + ($bodyHeight / 2.0)
    )
    Set-VisioShapeName -Shape $body -Name "Robot_Body"
    Set-VisioNodeStyle `
        -Shape $body `
        -FillColor ([string]$config.RobotBodyColor) `
        -LineColor ([string]$config.RobotBodyEdgeColor) `
        -LineWeightPt 0.85 `
        -RoundingIn 0.035
    Add-ShapeToVisioLayer -Layer $layersByName["Robot"] -Shape $body
    $robotParts += $body

    $radarDiameter = 0.095
    $robotRadar = $page.DrawOval(
        $centerX - ($radarDiameter / 2.0),
        $centerY - ($radarDiameter / 2.0),
        $centerX + ($radarDiameter / 2.0),
        $centerY + ($radarDiameter / 2.0)
    )
    Set-VisioShapeName -Shape $robotRadar -Name "Robot_Radar"
    Set-VisioNodeStyle `
        -Shape $robotRadar `
        -FillColor ([string]$config.RobotRadarColor) `
        -LineColor ([string]$config.RobotBodyEdgeColor) `
        -LineWeightPt 0.70
    Add-ShapeToVisioLayer -Layer $layersByName["Robot"] -Shape $robotRadar
    $robotParts += $robotRadar

    $heading = New-ArrowShape `
        -Page $page `
        -Layer $layersByName["Robot"] `
        -Name "Robot_Heading" `
        -BeginX $centerX `
        -BeginY ($centerY + 0.025) `
        -EndX $centerX `
        -EndY ($centerY + 0.105) `
        -LineColor "#F8F8F6" `
        -LineWeightPt 0.85 `
        -EndArrow 13 `
        -EndArrowSize 1
    $robotParts += $heading

    $robotGroup = New-VisioShapeGroup `
        -Page $page `
        -Shapes $robotParts `
        -Name "Robot_Group" `
        -Layer $layersByName["Robot"]
    foreach ($shape in $robotParts) {
        Release-VisioComObject -ComObject $shape
    }
    $robotParts = @()
    $robotGroup.BringToFront()
    Write-VisioStage -Stage "robot_group_created"

    $null = $document.SaveAs($outputPath)
    Write-VisioStage -Stage "document_saved"
    $pngResult = Export-VisioPng -Page $page -Path $pngPath -Dpi 300
    Write-VisioStage -Stage "png_exported"
    $document.Close()

    Release-VisioComObject -ComObject $radarGroup
    Release-VisioComObject -ComObject $radarBoundary
    Release-VisioComObject -ComObject $actionGroup
    Release-VisioComObject -ComObject $robotGroup
    $radarGroup = $null
    $radarBoundary = $null
    $actionGroup = $null
    $robotGroup = $null
    foreach ($layer in $layersByName.Values) {
        Release-VisioComObject -ComObject $layer
    }
    $layersByName = @{}
    Release-VisioComObject -ComObject $page
    $page = $null
    if ($null -ne $workspace) {
        $workspace.Page = $null
        $workspace.Document = $null
    }
    Release-VisioComObject -ComObject $document
    $document = $null
    Write-VisioStage -Stage "document_closed"

    if (-not (Test-Path -LiteralPath $outputPath)) {
        throw "Visio output was not created at '$outputPath'."
    }
    $outputFile = Get-Item -LiteralPath $outputPath
    if ([long]$outputFile.Length -le 0) {
        throw "Visio output exists but is empty: '$outputPath'."
    }

    Stop-VisioSession -Session $session
    $session = $null
    Write-VisioStage -Stage "generation_session_stopped"

    $validationOutput = & powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $MyInvocation.MyCommand.Path `
        -ValidateOnly `
        -ValidationPath $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent Visio reopen validation process exited with code $LASTEXITCODE."
    }
    $validationJson = $validationOutput -join [Environment]::NewLine
    $validation = $validationJson | ConvertFrom-Json
    Write-VisioStage -Stage "document_reopened_and_validated"

    $summaryPath = Join-Path `
        ([IO.Path]::GetTempPath()) `
        ("fig2_visio_validation_{0}_{1}.json" -f $PID, (Get-Date -Format "yyyyMMdd_HHmmss"))
    $validationJson | Set-Content -LiteralPath $summaryPath -Encoding UTF8

    $result = [ordered]@{
        OutputPath = $outputFile.FullName
        PngPath = $pngResult.Path
        FileSizeBytes = [long]$outputFile.Length
        PngFileSizeBytes = [long]$pngResult.FileSizeBytes
        PngPixelWidth = [int]$pngResult.PixelWidth
        PngPixelHeight = [int]$pngResult.PixelHeight
        PngDpi = [int]$pngResult.Dpi
        UsedTimestampFallback = [bool]$outputPair.UsedTimestampFallback
        LockedPaths = @($outputPair.LockedPaths)
        ReopenValidation = "passed"
        ValidationSummaryPath = $summaryPath
        Validation = $validation
    }
    $result | ConvertTo-Json -Depth 10
}
catch {
    throw "Failed to build or validate Figure 2 Visio document: $($_.Exception.Message)`n$($_.ScriptStackTrace)"
}
finally {
    if ($null -ne $document) {
        try {
            $document.Close()
        }
        catch {
        }
    }
    Release-VisioComObject -ComObject $radarGroup
    Release-VisioComObject -ComObject $radarBoundary
    Release-VisioComObject -ComObject $actionGroup
    Release-VisioComObject -ComObject $robotGroup
    foreach ($layer in $layersByName.Values) {
        Release-VisioComObject -ComObject $layer
    }
    Release-VisioComObject -ComObject $page
    if ($null -ne $workspace) {
        $workspace.Page = $null
        $workspace.Document = $null
    }
    Release-VisioComObject -ComObject $document
    $document = $null
    $page = $null
    $workspace = $null
    Stop-VisioSession -Session $session
}
