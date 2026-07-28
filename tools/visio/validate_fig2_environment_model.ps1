Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Add-Fig2ShapeInventoryRecord {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Shape,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [Collections.Generic.List[object]]$Records,
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

    $width = [double]$Shape.CellsU("Width").ResultIU
    $height = [double]$Shape.CellsU("Height").ResultIU
    $pinX = [double]$Shape.CellsU("PinX").ResultIU
    $pinY = [double]$Shape.CellsU("PinY").ResultIU
    $lineLength = 0.0
    if (
        [int]$Shape.CellExistsU("BeginX", 0) -ne 0 -and
        [int]$Shape.CellExistsU("BeginY", 0) -ne 0 -and
        [int]$Shape.CellExistsU("EndX", 0) -ne 0 -and
        [int]$Shape.CellExistsU("EndY", 0) -ne 0
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
    $hyperlinks = $null
    try {
        $hyperlinks = $Shape.Hyperlinks
        $hyperlinkCount = [int]$hyperlinks.Count
    }
    catch {
        $hyperlinkCount = 0
    }
    finally {
        Release-VisioComObject -ComObject $hyperlinks
    }

    $Records.Add([pscustomobject]@{
        Name = [string]$Shape.NameU
        ParentName = $ParentName
        Type = [int]$Shape.Type
        LayerNames = [string[]]$layerNames.ToArray()
        WidthIn = $width
        HeightIn = $height
        PinXIn = $pinX
        PinYIn = $pinY
        LineLengthIn = $lineLength
        HyperlinkCount = $hyperlinkCount
    })

    $children = $null
    try {
        $children = $Shape.Shapes
        for ($childIndex = 1; $childIndex -le [int]$children.Count; $childIndex++) {
            $child = $null
            try {
                $child = $children.Item($childIndex)
                Add-Fig2ShapeInventoryRecord `
                    -Shape $child `
                    -Records $Records `
                    -ParentName ([string]$Shape.NameU)
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
    param(
        [Parameter(Mandatory = $true)]
        $Page
    )

    $records = New-Object 'System.Collections.Generic.List[object]'
    $shapes = $null
    try {
        $shapes = $Page.Shapes
        for ($index = 1; $index -le [int]$shapes.Count; $index++) {
            $shape = $null
            try {
                $shape = $shapes.Item($index)
                Add-Fig2ShapeInventoryRecord -Shape $shape -Records $records
            }
            finally {
                Release-VisioComObject -ComObject $shape
            }
        }
    }
    finally {
        Release-VisioComObject -ComObject $shapes
    }
    return [object[]]$records.ToArray()
}

function Test-Fig2VisioDocument {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Document
    )

    $gridRows = 15
    $gridCols = 15
    $cellSizeIn = 0.40
    $centerX = 3.70
    $centerY = 3.40
    $expectedActionLengthIn = 1.50 * $cellSizeIn
    $expectedRayLengthIn = 5.15 * $cellSizeIn
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
        "Action_N",
        "Action_NE",
        "Action_E",
        "Action_SE",
        "Action_S",
        "Action_SW",
        "Action_W",
        "Action_NW"
    )
    $rayNames = 1..32 | ForEach-Object { "Radar_Ray_{0:D2}" -f $_ }

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

        $cellRecords = @(
            $inventory | Where-Object { $_.Name -match '^Cell_r\d{2}_c\d{2}$' }
        )
        if ($cellRecords.Count -ne ($gridRows * $gridCols)) {
            throw "Expected $($gridRows * $gridCols) occupancy cells, found $($cellRecords.Count)."
        }
        foreach ($cell in $cellRecords) {
            if ("Occupancy_Cells" -notin $cell.LayerNames) {
                throw "Cell '$($cell.Name)' is not on Occupancy_Cells."
            }
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

        $actionLengths = New-Object System.Collections.Generic.List[double]
        foreach ($actionName in $actionNames) {
            if (-not $shapeByName.ContainsKey($actionName)) {
                throw "Required action arrow '$actionName' is missing."
            }
            $record = $shapeByName[$actionName]
            if ("Action_Arrows" -notin $record.LayerNames) {
                throw "Action arrow '$actionName' is not on Action_Arrows."
            }
            $actionLengths.Add([double]$record.LineLengthIn)
            if ([Math]::Abs([double]$record.LineLengthIn - $expectedActionLengthIn) -gt (0.01 * $expectedActionLengthIn)) {
                throw "Action arrow '$actionName' length $($record.LineLengthIn) is outside the 1% tolerance."
            }
        }

        $rayLengths = New-Object System.Collections.Generic.List[double]
        foreach ($rayName in $rayNames) {
            if (-not $shapeByName.ContainsKey($rayName)) {
                throw "Required radar ray '$rayName' is missing."
            }
            $record = $shapeByName[$rayName]
            if ("Radar_Rays" -notin $record.LayerNames) {
                throw "Radar ray '$rayName' is not on Radar_Rays."
            }
            $rayLengths.Add([double]$record.LineLengthIn)
            if ([Math]::Abs([double]$record.LineLengthIn - $expectedRayLengthIn) -gt (0.01 * $expectedRayLengthIn)) {
                throw "Radar ray '$rayName' length $($record.LineLengthIn) is outside the 1% tolerance."
            }
        }

        $rayMin = [double](($rayLengths | Measure-Object -Minimum).Minimum)
        $rayMax = [double](($rayLengths | Measure-Object -Maximum).Maximum)
        $rayMean = [double](($rayLengths | Measure-Object -Average).Average)
        $rayRelativeDifference = if ($rayMean -gt 0.0) {
            ($rayMax - $rayMin) / $rayMean
        }
        else {
            0.0
        }
        if ($rayRelativeDifference -gt 0.01) {
            throw "Radar ray relative length difference exceeds 1%."
        }

        $robot = $shapeByName["Robot_Group"]
        $robotWidthRatio = [double]$robot.WidthIn / $cellSizeIn
        $robotHeightRatio = [double]$robot.HeightIn / $cellSizeIn
        if ($robotWidthRatio -gt 0.95 -or $robotHeightRatio -gt 0.95) {
            throw "Robot group exceeds the allowed one-cell visual envelope."
        }
        if (
            [Math]::Abs([double]$robot.PinXIn - $centerX) -gt 0.002 -or
            [Math]::Abs([double]$robot.PinYIn - $centerY) -gt 0.002
        ) {
            throw "Robot group is not centered on the center grid cell."
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
                $layerStates[$layerName] = [ordered]@{
                    Visible = $visible
                    Locked = $locked
                }
            }
            catch {
                throw "Required layer '$layerName' is missing or invalid: $($_.Exception.Message)"
            }
            finally {
                Release-VisioComObject -ComObject $layer
            }
            $layerShapeCounts[$layerName] = @(
                $inventory | Where-Object { $layerName -in $_.LayerNames }
            ).Count
            if ([int]$layerShapeCounts[$layerName] -lt 1) {
                throw "Layer '$layerName' is not used by any shape."
            }
        }

        $foreignObjectCount = @(
            $inventory | Where-Object { [int]$_.Type -eq 4 }
        ).Count
        $shapeHyperlinkCount = [int]((
            $inventory | Measure-Object -Property HyperlinkCount -Sum
        ).Sum)
        $dataRecordsetCount = 0
        try {
            $dataRecordsetCount = [int]$Document.DataRecordsets.Count
        }
        catch {
        }
        if ($foreignObjectCount -ne 0) {
            throw "Found $foreignObjectCount foreign image/OLE object(s); expected none."
        }
        if (($shapeHyperlinkCount + $dataRecordsetCount) -ne 0) {
            throw "Found hyperlinks or external data links; expected none."
        }

        $pageWidth = [double]$page.PageSheet.CellsU("PageWidth").ResultIU
        $pageHeight = [double]$page.PageSheet.CellsU("PageHeight").ResultIU
        return [pscustomobject]@{
            PageCount = [int]$Document.Pages.Count
            PageWidthIn = [Math]::Round($pageWidth, 3)
            PageHeightIn = [Math]::Round($pageHeight, 3)
            GridRows = $gridRows
            GridCols = $gridCols
            CellSizeIn = $cellSizeIn
            OccupancyCellCount = $cellRecords.Count
            TopLevelShapeCount = [int]$page.Shapes.Count
            TotalShapeCountIncludingGroupMembers = $inventory.Count
            RobotWidthIn = [Math]::Round([double]$robot.WidthIn, 4)
            RobotHeightIn = [Math]::Round([double]$robot.HeightIn, 4)
            RobotWidthCellRatio = [Math]::Round($robotWidthRatio, 4)
            RobotHeightCellRatio = [Math]::Round($robotHeightRatio, 4)
            RobotCenterXIn = [Math]::Round([double]$robot.PinXIn, 4)
            RobotCenterYIn = [Math]::Round([double]$robot.PinYIn, 4)
            ActionArrowCount = $actionNames.Count
            ActionLengthMinIn = [Math]::Round([double](($actionLengths | Measure-Object -Minimum).Minimum), 4)
            ActionLengthMaxIn = [Math]::Round([double](($actionLengths | Measure-Object -Maximum).Maximum), 4)
            ActionLengthMeanIn = [Math]::Round([double](($actionLengths | Measure-Object -Average).Average), 4)
            ActionLengthCells = [Math]::Round(
                [double](($actionLengths | Measure-Object -Average).Average) / $cellSizeIn,
                4
            )
            RadarRayCount = $rayNames.Count
            RadarRayLengthMinIn = [Math]::Round($rayMin, 4)
            RadarRayLengthMaxIn = [Math]::Round($rayMax, 4)
            RadarRayRelativeDifference = [Math]::Round($rayRelativeDifference, 8)
            RadarBoundaryCount = 1
            LayerCount = [int]$layers.Count
            LayerNames = $requiredLayerNames
            LayerShapeCounts = $layerShapeCounts
            LayerStates = $layerStates
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
