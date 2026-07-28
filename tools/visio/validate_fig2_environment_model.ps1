Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Add-Fig2ShapeInventoryRecord {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Shape,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][Collections.Generic.List[object]]$Records,
        [string]$ParentName = ""
    )

    $layerNames = New-Object System.Collections.Generic.List[string]
    for ($layerIndex = 1; $layerIndex -le [int]$Shape.LayerCount; $layerIndex++) {
        $layer = $null
        try {
            $layer = $Shape.Layer($layerIndex)
            $layerNames.Add([string]$layer.NameU)
        }
        finally {
            Release-VisioComObject -ComObject $layer
        }
    }
    $beginX = $null
    $beginY = $null
    $endX = $null
    $endY = $null
    $lineLength = 0.0
    if (
        [int]$Shape.CellExistsU("BeginX", 0) -ne 0 -and
        [int]$Shape.CellExistsU("EndX", 0) -ne 0
    ) {
        $beginX = [double]$Shape.CellsU("BeginX").ResultIU
        $beginY = [double]$Shape.CellsU("BeginY").ResultIU
        $endX = [double]$Shape.CellsU("EndX").ResultIU
        $endY = [double]$Shape.CellsU("EndY").ResultIU
        $lineLength = [Math]::Sqrt(
            (($endX - $beginX) * ($endX - $beginX)) +
            (($endY - $beginY) * ($endY - $beginY))
        )
    }
    $hyperlinkCount = 0
    try {
        $hyperlinkCount = [int]$Shape.Hyperlinks.Count
    }
    catch {
    }
    $fillFormula = if ([int]$Shape.CellExistsU("FillForegnd", 0) -ne 0) {
        [string]$Shape.CellsU("FillForegnd").FormulaU
    }
    else { "" }
    $lineFormula = if ([int]$Shape.CellExistsU("LineColor", 0) -ne 0) {
        [string]$Shape.CellsU("LineColor").FormulaU
    }
    else { "" }
    $lineWeight = if ([int]$Shape.CellExistsU("LineWeight", 0) -ne 0) {
        [double]$Shape.CellsU("LineWeight").ResultIU
    }
    else { 0.0 }
    $Records.Add([pscustomobject]@{
        Name = [string]$Shape.NameU
        ParentName = $ParentName
        Type = [int]$Shape.Type
        LayerNames = [string[]]$layerNames.ToArray()
        WidthIn = [double]$Shape.CellsU("Width").ResultIU
        HeightIn = [double]$Shape.CellsU("Height").ResultIU
        PinXIn = [double]$Shape.CellsU("PinX").ResultIU
        PinYIn = [double]$Shape.CellsU("PinY").ResultIU
        BeginXIn = $beginX
        BeginYIn = $beginY
        EndXIn = $endX
        EndYIn = $endY
        LineLengthIn = $lineLength
        FillFormula = $fillFormula
        LineFormula = $lineFormula
        LineWeightIn = $lineWeight
        HyperlinkCount = $hyperlinkCount
    })

    $children = $null
    try {
        $children = $Shape.Shapes
        for ($childIndex = 1; $childIndex -le [int]$children.Count; $childIndex++) {
            $child = $null
            try {
                $child = $children.Item($childIndex)
                Add-Fig2ShapeInventoryRecord -Shape $child -Records $Records -ParentName ([string]$Shape.NameU)
            }
            finally {
                Release-VisioComObject -ComObject $child
            }
        }
    }
    finally {
        Release-VisioComObject -ComObject $children
    }
}

function Get-Fig2ShapeInventory {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)]$Page)

    $records = New-Object 'System.Collections.Generic.List[object]'
    for ($index = 1; $index -le [int]$Page.Shapes.Count; $index++) {
        $shape = $null
        try {
            $shape = $Page.Shapes.Item($index)
            Add-Fig2ShapeInventoryRecord -Shape $shape -Records $records
        }
        finally {
            Release-VisioComObject -ComObject $shape
        }
    }
    return [object[]]$records.ToArray()
}

function Test-Fig2ColorFormula {
    param(
        [Parameter(Mandatory = $true)][string]$Actual,
        [Parameter(Mandatory = $true)][string]$HexColor,
        [Parameter(Mandatory = $true)][string]$Context
    )
    $expected = ConvertTo-VisioRgbFormula -HexColor $HexColor
    if (-not $Actual.ToUpperInvariant().Contains($expected.ToUpperInvariant())) {
        throw "$Context color '$Actual' does not match style contract '$expected'."
    }
}

function Test-Fig2VisioDocument {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)]$Blueprint,
        [Parameter(Mandatory = $true)]$Style
    )

    $gridRows = [int]$Blueprint.local_shape[0]
    $gridCols = [int]$Blueprint.local_shape[1]
    $centerRow = [int]$Blueprint.center_state[0]
    $centerCol = [int]$Blueprint.center_state[1]
    if ($gridRows -ne 21 -or $gridCols -ne 21) {
        throw "Figure 2 blueprint must be 21x21, got ${gridRows}x${gridCols}."
    }
    if ([int]$Blueprint.scan_radius -ne 10) {
        throw "Figure 2 blueprint must use scan_radius=10."
    }
    $requiredLayerNames = @(
        "Grid_Background",
        "Occupancy_Cells",
        "Grid_Lines",
        "Radar_Rays",
        "Radar_Boundary",
        "Action_Arrows",
        "Robot"
    )
    $actionNames = @(
        "Action_N", "Action_NE", "Action_E", "Action_SE",
        "Action_S", "Action_SW", "Action_W", "Action_NW"
    )
    if ([int]$Document.Pages.Count -ne 1) {
        throw "Expected one page, found $($Document.Pages.Count)."
    }

    $page = $null
    $layers = $null
    try {
        $page = $Document.Pages.Item(1)
        $inventory = @(Get-Fig2ShapeInventory -Page $page)
        $shapeByName = @{}
        foreach ($record in $inventory) {
            if ($shapeByName.ContainsKey($record.Name)) {
                throw "Duplicate Visio shape name '$($record.Name)'."
            }
            $shapeByName[$record.Name] = $record
        }
        $cellRecords = @($inventory | Where-Object { $_.Name -match '^Cell_r\d{2}_c\d{2}$' })
        if ($cellRecords.Count -ne ($gridRows * $gridCols)) {
            throw "Expected $($gridRows * $gridCols) editable occupancy cells, found $($cellRecords.Count)."
        }
        $cellSizeIn = [double]$cellRecords[0].WidthIn
        if ($cellSizeIn -le 0.0) {
            throw "Invalid local-grid cell size."
        }
        foreach ($cell in $cellRecords) {
            if ("Occupancy_Cells" -notin $cell.LayerNames) {
                throw "Cell '$($cell.Name)' is not on Occupancy_Cells."
            }
            if (
                [Math]::Abs([double]$cell.WidthIn - $cellSizeIn) -gt 0.00001 -or
                [Math]::Abs([double]$cell.HeightIn - $cellSizeIn) -gt 0.00001
            ) {
                throw "Cell '$($cell.Name)' does not use the shared grid scale."
            }
            if ($cell.Name -notmatch '^Cell_r(\d{2})_c(\d{2})$') {
                throw "Unexpected cell name '$($cell.Name)'."
            }
            $row = [int]$Matches[1]
            $col = [int]$Matches[2]
            $state = [int]$Blueprint.local_snap_t[$row][$col]
            $color = switch ($state) {
                -1 { [string]$Style.OccupancyPalette.unknown }
                0 { [string]$Style.OccupancyPalette.free }
                1 { [string]$Style.OccupancyPalette.obstacle }
                default { throw "Unexpected local_snap value $state at ($row,$col)." }
            }
            Test-Fig2ColorFormula -Actual ([string]$cell.FillFormula) -HexColor $color -Context $cell.Name
        }
        foreach ($requiredName in @(
            "Local_Grid_Background",
            "Radar_Range_Boundary",
            "Radar_Rays_Group",
            "Action_Arrows_Group",
            "Robot_Group"
        )) {
            if (-not $shapeByName.ContainsKey($requiredName)) {
                throw "Required Visio shape '$requiredName' is missing."
            }
        }

        $centerCellName = "Cell_r{0:D2}_c{1:D2}" -f $centerRow, $centerCol
        $centerCell = $shapeByName[$centerCellName]
        $centerX = [double]$centerCell.PinXIn
        $centerY = [double]$centerCell.PinYIn
        $expectedActionLengthIn = 1.50 * $cellSizeIn
        $actionLengths = New-Object System.Collections.Generic.List[double]
        $actionColorFormulas = New-Object System.Collections.Generic.List[string]
        foreach ($actionName in $actionNames) {
            if (-not $shapeByName.ContainsKey($actionName)) {
                throw "Required action arrow '$actionName' is missing."
            }
            $record = $shapeByName[$actionName]
            if ("Action_Arrows" -notin $record.LayerNames) {
                throw "Action arrow '$actionName' is not on Action_Arrows."
            }
            if ([Math]::Abs([double]$record.LineLengthIn - $expectedActionLengthIn) -gt (0.01 * $expectedActionLengthIn)) {
                throw "Action arrow '$actionName' is outside the 1% length tolerance."
            }
            Test-Fig2ColorFormula `
                -Actual ([string]$record.LineFormula) `
                -HexColor ([string]$Style.RadarPalette.action) `
                -Context $actionName
            $actionLengths.Add([double]$record.LineLengthIn)
            $actionColorFormulas.Add([string]$record.LineFormula)
        }
        if (@($actionColorFormulas | Sort-Object -Unique).Count -ne 1) {
            throw "Figure 2 action arrows must use one uniform color."
        }

        $rayLengths = New-Object System.Collections.Generic.List[double]
        $rayCount = [int]$Blueprint.representative_rays.Count
        for ($index = 0; $index -lt $rayCount; $index++) {
            $rayName = "Radar_Ray_{0:D2}" -f ($index + 1)
            if (-not $shapeByName.ContainsKey($rayName)) {
                throw "Required radar ray '$rayName' is missing."
            }
            $record = $shapeByName[$rayName]
            if ("Radar_Rays" -notin $record.LayerNames) {
                throw "Radar ray '$rayName' is not on Radar_Rays."
            }
            Test-Fig2ColorFormula `
                -Actual ([string]$record.LineFormula) `
                -HexColor ([string]$Style.RadarPalette.ray) `
                -Context $rayName
            $endRow = [int]$Blueprint.representative_rays[$index].end_local_rc[0]
            $endCol = [int]$Blueprint.representative_rays[$index].end_local_rc[1]
            $expectedRayLength = [Math]::Sqrt(
                (($endCol - $centerCol) * ($endCol - $centerCol)) +
                (($endRow - $centerRow) * ($endRow - $centerRow))
            ) * $cellSizeIn
            if ([Math]::Abs([double]$record.LineLengthIn - $expectedRayLength) -gt (0.01 * $cellSizeIn)) {
                throw "Radar ray '$rayName' length does not match its truncated blueprint endpoint."
            }
            $terminalState = [int]$Blueprint.local_snap_t[$endRow][$endCol]
            if ($terminalState -eq -1) {
                throw "Radar ray '$rayName' ends in an invisible cell."
            }
            $rayLengths.Add([double]$record.LineLengthIn)
        }
        $rayMin = [double](($rayLengths | Measure-Object -Minimum).Minimum)
        $rayMax = [double](($rayLengths | Measure-Object -Maximum).Maximum)
        if (($rayMax - $rayMin) -lt (0.25 * $cellSizeIn)) {
            throw "Radar rays were unexpectedly forced to equal length."
        }

        $robot = $shapeByName["Robot_Group"]
        $robotWidthRatio = [double]$robot.WidthIn / $cellSizeIn
        $robotHeightRatio = [double]$robot.HeightIn / $cellSizeIn
        if ($robotWidthRatio -gt 0.95 -or $robotHeightRatio -gt 0.95) {
            throw "Robot group exceeds the 0.95-cell visual envelope."
        }
        if (
            [Math]::Abs([double]$robot.PinXIn - $centerX) -gt 0.002 -or
            [Math]::Abs([double]$robot.PinYIn - $centerY) -gt 0.002
        ) {
            throw "Robot group is not centered on sensor.center_state."
        }
        foreach ($robotPart in @(
            @("Robot_body", "FillFormula", [string]$Style.RobotPalette.body),
            @("Robot_radar", "FillFormula", [string]$Style.RobotPalette.radar),
            @("Robot_heading", "LineFormula", [string]$Style.RobotPalette.heading)
        )) {
            if (-not $shapeByName.ContainsKey($robotPart[0])) {
                throw "Robot part '$($robotPart[0])' is missing."
            }
            Test-Fig2ColorFormula `
                -Actual ([string]$shapeByName[$robotPart[0]].($robotPart[1])) `
                -HexColor ([string]$robotPart[2]) `
                -Context ([string]$robotPart[0])
        }
        $wheelRecords = @($inventory | Where-Object { $_.Name -match '^Robot_wheel_\d{2}$' })
        if ($wheelRecords.Count -ne 4) {
            throw "Expected four robot wheels, found $($wheelRecords.Count)."
        }
        foreach ($wheel in $wheelRecords) {
            Test-Fig2ColorFormula `
                -Actual ([string]$wheel.FillFormula) `
                -HexColor ([string]$Style.RobotPalette.wheels) `
                -Context ([string]$wheel.Name)
        }

        $layerShapeCounts = [ordered]@{}
        $layerStates = [ordered]@{}
        $layers = $page.Layers
        foreach ($layerName in $requiredLayerNames) {
            $layer = $null
            try {
                $layer = $layers.ItemU($layerName)
                $visible = [int]$layer.CellsC(4).ResultIU
                $locked = [int]$layer.CellsC(7).ResultIU
                if ($visible -ne 1 -or $locked -ne 0) {
                    throw "Layer '$layerName' must be visible and unlocked."
                }
                $layerStates[$layerName] = [ordered]@{ Visible = $visible; Locked = $locked }
            }
            finally {
                Release-VisioComObject -ComObject $layer
            }
            $layerShapeCounts[$layerName] = @(
                $inventory | Where-Object { $layerName -in $_.LayerNames }
            ).Count
            if ([int]$layerShapeCounts[$layerName] -lt 1) {
                throw "Layer '$layerName' is unused."
            }
        }
        $foreignObjectCount = @($inventory | Where-Object { [int]$_.Type -eq 4 }).Count
        $shapeHyperlinkCount = [int](($inventory | Measure-Object -Property HyperlinkCount -Sum).Sum)
        $dataRecordsetCount = 0
        try { $dataRecordsetCount = [int]$Document.DataRecordsets.Count } catch {}
        if ($foreignObjectCount -ne 0) {
            throw "Found $foreignObjectCount foreign image/OLE object(s); expected none."
        }
        if (($shapeHyperlinkCount + $dataRecordsetCount) -ne 0) {
            throw "Found hyperlinks or external data links; expected none."
        }
        return [pscustomobject]@{
            PageCount = [int]$Document.Pages.Count
            PageWidthIn = [Math]::Round([double]$page.PageSheet.CellsU("PageWidth").ResultIU, 3)
            PageHeightIn = [Math]::Round([double]$page.PageSheet.CellsU("PageHeight").ResultIU, 3)
            Seed = [int]$Blueprint.seed
            Step = [int]$Blueprint.step
            ScanRadius = [int]$Blueprint.scan_radius
            GridRows = $gridRows
            GridCols = $gridCols
            CenterState = @($centerRow, $centerCol)
            CellSizeIn = [Math]::Round($cellSizeIn, 4)
            OccupancyCellCount = $cellRecords.Count
            RadarRayCount = $rayCount
            RadarRayLengthMinIn = [Math]::Round($rayMin, 4)
            RadarRayLengthMaxIn = [Math]::Round($rayMax, 4)
            RadarRaysVariableLength = $true
            ActionArrowCount = $actionNames.Count
            ActionColorsUniform = $true
            RobotWidthCellRatio = [Math]::Round($robotWidthRatio, 4)
            RobotHeightCellRatio = [Math]::Round($robotHeightRatio, 4)
            RobotWheelCount = $wheelRecords.Count
            StyleContractPath = [string]$Style.ContractPath
            StyleContractVersion = [string]$Style.Version
            LayerCount = [int]$layers.Count
            LayerNames = $requiredLayerNames
            LayerShapeCounts = $layerShapeCounts
            LayerStates = $layerStates
            TopLevelShapeCount = [int]$page.Shapes.Count
            TotalShapeCountIncludingGroupMembers = $inventory.Count
            ForeignObjectCount = $foreignObjectCount
            OleObjectCount = $foreignObjectCount
            ExternalLinkCount = [int]($shapeHyperlinkCount + $dataRecordsetCount)
        }
    }
    finally {
        Release-VisioComObject -ComObject $layers
        Release-VisioComObject -ComObject $page
    }
}
