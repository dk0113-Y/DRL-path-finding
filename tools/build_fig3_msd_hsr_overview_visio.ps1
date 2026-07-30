[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AssetsDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputVsdx,
    [Parameter(Mandatory = $true)]
    [string]$OutputSvg,
    [Parameter(Mandatory = $true)]
    [string]$OutputPng,
    [int]$Dpi = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonPath = Join-Path $PSScriptRoot "visio\visio_common.ps1"
. $commonPath

$script:FontName = "Arial"
$script:Ink = "#19324A"
$script:ModuleLine = "#7E91A4"
$script:ModuleFill = "#FCFDFE"
$script:Warm = "#C96144"
$script:WarmLight = "#FBF0EA"
$script:Cool = "#5185C0"
$script:CoolLight = "#EEF4FA"
$script:Green = "#55966B"
$script:GreenLight = "#EEF7F0"
$script:Neutral = "#607487"
$script:NeutralLight = "#F4F7F9"

function Resolve-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [IO.Path]::GetFullPath($Path)
}

function Add-Fig3Rect {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][double]$Left,
        [Parameter(Mandatory = $true)][double]$Bottom,
        [Parameter(Mandatory = $true)][double]$Right,
        [Parameter(Mandatory = $true)][double]$Top,
        [Parameter(Mandatory = $true)][string]$Fill,
        [Parameter(Mandatory = $true)][string]$Line,
        [double]$LineWeightPt = 1.0,
        [double]$RoundingIn = 0.06,
        [string]$Text = "",
        [double]$FontSizePt = 6.5,
        [bool]$Bold = $false,
        [string]$TextColor = ""
    )
    $shape = $Page.DrawRectangle($Left, $Bottom, $Right, $Top)
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioNodeStyle `
        -Shape $shape `
        -FillColor $Fill `
        -LineColor $Line `
        -LineWeightPt $LineWeightPt `
        -RoundingIn $RoundingIn
    if (-not [string]::IsNullOrWhiteSpace($Text)) {
        $shape.Text = $Text
        $color = if ([string]::IsNullOrWhiteSpace($TextColor)) { $script:Ink } else { $TextColor }
        Set-VisioTextFormat `
            -Shape $shape `
            -FontName $script:FontName `
            -FontSizePt $FontSizePt `
            -HorizontalAlign "Center" `
            -VerticalAlign "Middle" `
            -TextColor $color
        if ($Bold) {
            Set-VisioCellFormula -Shape $shape -CellName "Char.Style" -Formula "1"
        }
        foreach ($marginName in @("LeftMargin", "RightMargin", "TopMargin", "BottomMargin")) {
            Set-VisioCellFormula -Shape $shape -CellName $marginName -Formula "0.01 in" -Optional
        }
    }
    return ,$shape
}

function Add-Fig3Text {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][double]$Width,
        [Parameter(Mandatory = $true)][double]$Height,
        [double]$FontSizePt = 6.5,
        [bool]$Bold = $false,
        [string]$Color = ""
    )
    $shape = Add-VisioTextBlock `
        -Page $Page `
        -Name $Name `
        -Text $Text `
        -CenterX $CenterX `
        -CenterY $CenterY `
        -Width $Width `
        -Height $Height `
        -FontName $script:FontName `
        -FontSizePt $FontSizePt `
        -TextColor $(if ([string]::IsNullOrWhiteSpace($Color)) { $script:Ink } else { $Color })
    if ($Bold) {
        Set-VisioCellFormula -Shape $shape -CellName "Char.Style" -Formula "1"
    }
    foreach ($marginName in @("LeftMargin", "RightMargin", "TopMargin", "BottomMargin")) {
        Set-VisioCellFormula -Shape $shape -CellName $marginName -Formula "0.005 in" -Optional
    }
    return ,$shape
}

function Add-Fig3Line {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][double]$BeginX,
        [Parameter(Mandatory = $true)][double]$BeginY,
        [Parameter(Mandatory = $true)][double]$EndX,
        [Parameter(Mandatory = $true)][double]$EndY,
        [Parameter(Mandatory = $true)][string]$Color,
        [double]$WeightPt = 1.0,
        [bool]$Arrow = $false,
        [bool]$Dashed = $false
    )
    $shape = $Page.DrawLine($BeginX, $BeginY, $EndX, $EndY)
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioLineStyle `
        -Shape $shape `
        -LineColor $Color `
        -LineWeightPt $WeightPt `
        -EndArrow $(if ($Arrow) { 13 } else { 0 }) `
        -EndArrowSize 2
    if ($Dashed) {
        Set-VisioCellFormula -Shape $shape -CellName "LinePattern" -Formula "2"
    }
    return ,$shape
}

function Add-Fig3Svg {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][double]$MaxWidth,
        [Parameter(Mandatory = $true)][double]$MaxHeight
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required SVG asset is missing: $Path"
    }
    $shape = $Page.Import($Path)
    Set-VisioShapeName -Shape $shape -Name $Name
    $sourceWidth = [Math]::Max(0.0001, [double]$shape.CellsU("Width").ResultIU)
    $sourceHeight = [Math]::Max(0.0001, [double]$shape.CellsU("Height").ResultIU)
    $scale = [Math]::Min($MaxWidth / $sourceWidth, $MaxHeight / $sourceHeight)
    $shape.CellsU("Width").ResultIUForce = $sourceWidth * $scale
    $shape.CellsU("Height").ResultIUForce = $sourceHeight * $scale
    $shape.CellsU("PinX").ResultIUForce = $CenterX
    $shape.CellsU("PinY").ResultIUForce = $CenterY
    Set-VisioCellFormula -Shape $shape -CellName "LockAspect" -Formula "1" -Optional
    return ,$shape
}

function Add-Fig3FeatureBlocks {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Prefix,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][string]$Color,
        [Parameter(Mandatory = $true)][string]$LightColor,
        [ValidateSet("Encoder", "Vector")]
        [string]$Mode = "Encoder"
    )
    if ($Mode -eq "Encoder") {
        $widths = @(0.12, 0.105, 0.09, 0.07)
        $heights = @(0.34, 0.28, 0.22, 0.15)
        $offsets = @(-0.19, -0.055, 0.065, 0.17)
        for ($index = 0; $index -lt 4; $index++) {
            $null = Add-Fig3Rect `
                -Page $Page `
                -Name "${Prefix}_block_$($index + 1)" `
                -Left ($CenterX + $offsets[$index] - ($widths[$index] / 2.0)) `
                -Bottom ($CenterY - ($heights[$index] / 2.0)) `
                -Right ($CenterX + $offsets[$index] + ($widths[$index] / 2.0)) `
                -Top ($CenterY + ($heights[$index] / 2.0)) `
                -Fill $LightColor `
                -Line $Color `
                -LineWeightPt 0.75 `
                -RoundingIn 0.01
        }
    }
    else {
        for ($index = 0; $index -lt 5; $index++) {
            $left = $CenterX - 0.23 + (0.09 * $index)
            $null = Add-Fig3Rect `
                -Page $Page `
                -Name "${Prefix}_value_$($index + 1)" `
                -Left $left `
                -Bottom ($CenterY - 0.055) `
                -Right ($left + 0.065) `
                -Top ($CenterY + 0.055) `
                -Fill $(if ($index -eq 4) { "#FFFFFF" } else { $LightColor }) `
                -Line $Color `
                -LineWeightPt 0.65 `
                -RoundingIn 0.008
        }
    }
}

function Add-Fig3QMatrix {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY
    )
    $labels = @(
        @("NW", "N", "NE"),
        @("W", "", "E"),
        @("SW", "S", "SE")
    )
    $cell = 0.18
    $left = $CenterX - (1.5 * $cell)
    $top = $CenterY + (1.5 * $cell)
    for ($row = 0; $row -lt 3; $row++) {
        for ($col = 0; $col -lt 3; $col++) {
            $label = [string]$labels[$row][$col]
            $cellLeft = $left + ($col * $cell)
            $cellTop = $top - ($row * $cell)
            $null = Add-Fig3Rect `
                -Page $Page `
                -Name "q_cell_${row}_${col}" `
                -Left $cellLeft `
                -Bottom ($cellTop - $cell) `
                -Right ($cellLeft + $cell) `
                -Top $cellTop `
                -Fill $(if ([string]::IsNullOrEmpty($label)) { "#FFFFFF" } else { $script:GreenLight }) `
                -Line $script:Green `
                -LineWeightPt 0.65 `
                -RoundingIn 0.0 `
                -Text $label `
                -FontSizePt 4.2 `
                -TextColor $script:Ink
        }
    }
}

function Add-Fig3MaskGlyph {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY
    )
    $valid = @(
        @($true, $true, $true),
        @($true, $false, $true),
        @($false, $true, $true)
    )
    $cell = 0.075
    $left = $CenterX - (1.5 * $cell)
    $top = $CenterY + (1.5 * $cell)
    for ($row = 0; $row -lt 3; $row++) {
        for ($col = 0; $col -lt 3; $col++) {
            $cellLeft = $left + ($col * $cell)
            $cellTop = $top - ($row * $cell)
            $null = Add-Fig3Rect `
                -Page $Page `
                -Name "mask_cell_${row}_${col}" `
                -Left $cellLeft `
                -Bottom ($cellTop - $cell) `
                -Right ($cellLeft + $cell) `
                -Top $cellTop `
                -Fill $(if ($valid[$row][$col]) { "#99C290" } else { "#FFFFFF" }) `
                -Line "#789486" `
                -LineWeightPt 0.45 `
                -RoundingIn 0.0
        }
    }
}

function Get-RecursiveShapeCount {
    param([Parameter(Mandatory = $true)]$Shapes)
    $count = 0
    for ($index = 1; $index -le [int]$Shapes.Count; $index++) {
        $shape = $Shapes.Item($index)
        try {
            $count++
            if ([int]$shape.Type -eq 2) {
                $count += Get-RecursiveShapeCount -Shapes $shape.Shapes
            }
        }
        finally {
            Release-VisioComObject -ComObject $shape
        }
    }
    return $count
}

$assetNames = @(
    "dynamic_cumulative_belief_map",
    "robot_position",
    "interaction_history",
    "local_occupancy_behavior_state",
    "global_hierarchical_semantic_state"
)
$assetsPath = Resolve-AbsolutePath -Path $AssetsDir
$assetPaths = @{}
foreach ($assetName in $assetNames) {
    $assetPath = Join-Path $assetsPath "${assetName}.svg"
    if (-not (Test-Path -LiteralPath $assetPath)) {
        throw "Required Figure 3 asset is missing: $assetPath"
    }
    $assetPaths[$assetName] = $assetPath
}
$assetManifestPath = Join-Path $assetsPath "fig3_overview_assets_manifest.json"
if (-not (Test-Path -LiteralPath $assetManifestPath)) {
    throw "Required Figure 3 asset manifest is missing: $assetManifestPath"
}
$assetManifest = Get-Content -LiteralPath $assetManifestPath -Raw | ConvertFrom-Json
$displayedBlockCount = [int]$assetManifest.displayed_unknown_block_count
$displayedEntryCount = [int]$assetManifest.displayed_frontier_entrance_count
$totalEntryCount = [int]$assetManifest.total_frontier_entrance_count

$vsdxPath = Resolve-AbsolutePath -Path $OutputVsdx
$svgPath = Resolve-AbsolutePath -Path $OutputSvg
$pngPath = Resolve-AbsolutePath -Path $OutputPng
foreach ($path in @($vsdxPath, $svgPath, $pngPath)) {
    $directory = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $directory)) {
        $null = New-Item -ItemType Directory -Path $directory
    }
}

$session = $null
$document = $null
$page = $null
$reopenedDocument = $null
$reopenedPage = $null
try {
    $session = Start-VisioSession -Visible $false
    $bundle = New-VisioDocumentPage `
        -Session $session `
        -PageWidthIn 7.20 `
        -PageHeightIn 3.08 `
        -PageName "Fig3_MSD_HSR_Overview"
    $document = $bundle.Document
    $page = $bundle.Page

    # Main module frames and titles.
    $null = Add-Fig3Rect -Page $page -Name "module_1_frame" -Left 0.98 -Bottom 0.05 -Right 2.47 -Top 3.03 -Fill $script:ModuleFill -Line $script:ModuleLine -LineWeightPt 1.0 -RoundingIn 0.08
    $null = Add-Fig3Rect -Page $page -Name "module_2_frame" -Left 2.54 -Bottom 0.05 -Right 4.62 -Top 3.03 -Fill $script:ModuleFill -Line $script:ModuleLine -LineWeightPt 1.0 -RoundingIn 0.08
    $null = Add-Fig3Rect -Page $page -Name "module_3_frame" -Left 4.69 -Bottom 0.05 -Right 7.17 -Top 3.03 -Fill $script:ModuleFill -Line $script:ModuleLine -LineWeightPt 1.0 -RoundingIn 0.08
    $null = Add-Fig3Text -Page $page -Name "module_1_title" -Text "Module 1`nExploration Information`nOrganization" -CenterX 1.725 -CenterY 2.76 -Width 1.36 -Height 0.48 -FontSizePt 7.2 -Bold $true
    $null = Add-Fig3Text -Page $page -Name "module_2_title" -Text "Module 2`nSemantic Feature Encoding" -CenterX 3.58 -CenterY 2.80 -Width 1.82 -Height 0.34 -FontSizePt 7.5 -Bold $true
    $null = Add-Fig3Text -Page $page -Name "module_3_title" -Text "Module 3`nAction-Value Estimation and Decision" -CenterX 5.93 -CenterY 2.80 -Width 2.20 -Height 0.34 -FontSizePt 7.5 -Bold $true

    # Three source cards.
    $inputCards = @(
        @{ Name = "input_dynamic_map"; Y0 = 1.99; Y1 = 2.64; Label = "Dynamic cumulative`nbelief map"; Asset = "dynamic_cumulative_belief_map"; W = 0.42; H = 0.38 },
        @{ Name = "input_robot_position"; Y0 = 1.22; Y1 = 1.87; Label = "Robot position"; Asset = "robot_position"; W = 0.37; H = 0.42 },
        @{ Name = "input_interaction_history"; Y0 = 0.45; Y1 = 1.10; Label = "Interaction history"; Asset = "interaction_history"; W = 0.37; H = 0.42 }
    )
    foreach ($card in $inputCards) {
        $null = Add-Fig3Rect -Page $page -Name ([string]$card.Name) -Left 0.03 -Bottom ([double]$card.Y0) -Right 0.88 -Top ([double]$card.Y1) -Fill "#FFFFFF" -Line "#9AA9B6" -LineWeightPt 0.9 -RoundingIn 0.06
        $centerY = ([double]$card.Y0 + [double]$card.Y1) / 2.0
        $null = Add-Fig3Text -Page $page -Name "$($card.Name)_label" -Text ([string]$card.Label) -CenterX 0.455 -CenterY ([double]$card.Y1 - 0.12) -Width 0.74 -Height 0.18 -FontSizePt 5.5
        $null = Add-Fig3Svg -Page $page -Name "asset_$($card.Asset)_input" -Path ([string]$assetPaths[[string]$card.Asset]) -CenterX 0.455 -CenterY ($centerY - 0.09) -MaxWidth $(if ([string]$card.Asset -eq "dynamic_cumulative_belief_map") { 0.56 } else { [double]$card.W }) -MaxHeight $(if ([string]$card.Asset -eq "dynamic_cumulative_belief_map") { 0.30 } else { [double]$card.H })
    }

    # Module 1 state organization.
    $null = Add-Fig3Rect -Page $page -Name "local_state_frame" -Left 1.10 -Bottom 1.67 -Right 2.34 -Top 2.55 -Fill $script:WarmLight -Line $script:Warm -LineWeightPt 1.0 -RoundingIn 0.06
    $null = Add-Fig3Text -Page $page -Name "local_state_label" -Text "Local`noccupancy–behavior`nstate" -CenterX 2.01 -CenterY 2.13 -Width 0.60 -Height 0.42 -FontSizePt 5.1 -Bold $true -Color $script:Warm
    $null = Add-Fig3Svg -Page $page -Name "asset_local_state_module1" -Path ([string]$assetPaths["local_occupancy_behavior_state"]) -CenterX 1.43 -CenterY 2.10 -MaxWidth 0.53 -MaxHeight 0.55

    $null = Add-Fig3Rect -Page $page -Name "global_state_frame" -Left 1.10 -Bottom 0.39 -Right 2.34 -Top 1.34 -Fill $script:CoolLight -Line $script:Cool -LineWeightPt 1.0 -RoundingIn 0.06
    $null = Add-Fig3Text -Page $page -Name "global_state_label" -Text "Global hierarchical`nsemantic state" -CenterX 2.01 -CenterY 0.86 -Width 0.60 -Height 0.38 -FontSizePt 5.1 -Bold $true -Color "#315F91"
    $null = Add-Fig3Svg -Page $page -Name "asset_global_state_module1" -Path ([string]$assetPaths["global_hierarchical_semantic_state"]) -CenterX 1.43 -CenterY 0.86 -MaxWidth 0.56 -MaxHeight 0.57
    if ($displayedBlockCount -gt 0) {
        $null = Add-Fig3Rect -Page $page -Name "semantic_overlay_parent_1" -Left 1.27 -Bottom 1.07 -Right 1.59 -Top 1.20 -Fill "#C8D9EB" -Line "#315F91" -LineWeightPt 0.8 -RoundingIn 0.025
        $overlayEntryCount = [Math]::Min(4, $displayedEntryCount)
        if ($overlayEntryCount -gt 0) {
            $entryCenters = if ($overlayEntryCount -eq 1) {
                @(1.43)
            }
            else {
                0..($overlayEntryCount - 1) | ForEach-Object {
                    1.20 + (0.15 * [double]$_)
                }
            }
            for ($entryIndex = 0; $entryIndex -lt $overlayEntryCount; $entryIndex++) {
                $entryX = [double]$entryCenters[$entryIndex]
                $null = Add-Fig3Line -Page $page -Name "semantic_overlay_edge_$($entryIndex + 1)" -BeginX 1.43 -BeginY 1.07 -EndX $entryX -EndY 0.67 -Color $script:Cool -WeightPt 0.65
                $null = Add-Fig3Rect -Page $page -Name "semantic_overlay_child_$($entryIndex + 1)" -Left ($entryX - 0.047) -Bottom 0.57 -Right ($entryX + 0.047) -Top 0.67 -Fill "#E8F0F8" -Line $script:Cool -LineWeightPt 0.7 -RoundingIn 0.012
            }
        }
        if ($totalEntryCount -gt $displayedEntryCount) {
            foreach ($dotX in @(1.80, 1.84, 1.88)) {
                $dot = $page.DrawOval($dotX, 0.605, $dotX + 0.018, 0.623)
                Set-VisioShapeName -Shape $dot -Name "semantic_overlay_ellipsis_$dotX"
                Set-VisioNodeStyle -Shape $dot -FillColor "#315F91" -LineColor "#315F91" -LineWeightPt 0.1
            }
        }
    }

    $fixed = Add-Fig3Rect -Page $page -Name "fixed_dimensional_representation" -Left 1.76 -Bottom 1.38 -Right 2.39 -Top 1.64 -Fill "#F8FCF8" -Line $script:Green -LineWeightPt 0.8 -RoundingIn 0.04 -Text "Fixed-dimensional`nrepresentation" -FontSizePt 5.5 -TextColor $script:Ink
    Set-VisioCellFormula -Shape $fixed -CellName "LinePattern" -Formula "2"

    # Accurate source dependencies: map and position share a junction; history only enters local state.
    $junction = $page.DrawOval(0.915, 1.515, 0.965, 1.565)
    Set-VisioShapeName -Shape $junction -Name "junction_spatial_inputs"
    Set-VisioNodeStyle -Shape $junction -FillColor "#FFFFFF" -LineColor $script:Neutral -LineWeightPt 0.8
    $null = Add-Fig3Line -Page $page -Name "connector_map_to_spatial_junction" -BeginX 0.88 -BeginY 2.315 -EndX 0.94 -EndY 1.54 -Color $script:Neutral -WeightPt 0.9
    $null = Add-Fig3Line -Page $page -Name "connector_position_to_spatial_junction" -BeginX 0.88 -BeginY 1.545 -EndX 0.94 -EndY 1.54 -Color $script:Neutral -WeightPt 0.9
    $null = Add-Fig3Line -Page $page -Name "connector_spatial_to_local" -BeginX 0.94 -BeginY 1.54 -EndX 1.10 -EndY 2.02 -Color $script:Warm -WeightPt 1.1 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_spatial_to_global" -BeginX 0.94 -BeginY 1.54 -EndX 1.10 -EndY 0.88 -Color $script:Cool -WeightPt 1.1 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_history_to_local" -BeginX 0.88 -BeginY 0.775 -EndX 1.10 -EndY 1.80 -Color $script:Warm -WeightPt 1.1 -Arrow $true

    # Module 2 upper local branch.
    $null = Add-Fig3Rect -Page $page -Name "module2_local_input" -Left 2.64 -Bottom 1.76 -Right 3.08 -Top 2.48 -Fill $script:WarmLight -Line $script:Warm -LineWeightPt 0.9 -RoundingIn 0.05
    $null = Add-Fig3Text -Page $page -Name "module2_local_input_label" -Text "Local`noccupancy–`nbehavior`nstate" -CenterX 2.86 -CenterY 2.30 -Width 0.50 -Height 0.36 -FontSizePt 5.0
    $null = Add-Fig3Svg -Page $page -Name "asset_local_state_module2" -Path ([string]$assetPaths["local_occupancy_behavior_state"]) -CenterX 2.86 -CenterY 1.91 -MaxWidth 0.28 -MaxHeight 0.28
    $null = Add-Fig3Rect -Page $page -Name "local_advantage_branch" -Left 3.23 -Bottom 1.76 -Right 3.81 -Top 2.48 -Fill $script:WarmLight -Line $script:Warm -LineWeightPt 0.9 -RoundingIn 0.05
    $null = Add-Fig3Text -Page $page -Name "local_advantage_branch_label" -Text "Local advantage`nbranch" -CenterX 3.52 -CenterY 2.35 -Width 0.50 -Height 0.22 -FontSizePt 5.6 -Bold $true
    Add-Fig3FeatureBlocks -Page $page -Prefix "local_encoder" -CenterX 3.52 -CenterY 1.99 -Color $script:Warm -LightColor "#F2CB9F" -Mode "Encoder"
    $null = Add-Fig3Rect -Page $page -Name "action_conditioned_features" -Left 3.96 -Bottom 1.76 -Right 4.52 -Top 2.48 -Fill $script:WarmLight -Line $script:Warm -LineWeightPt 0.9 -RoundingIn 0.05
    $null = Add-Fig3Text -Page $page -Name "action_conditioned_features_label" -Text "Action-`nconditioned`nlocal features" -CenterX 4.24 -CenterY 2.32 -Width 0.50 -Height 0.30 -FontSizePt 5.0
    Add-Fig3FeatureBlocks -Page $page -Prefix "local_feature_vector" -CenterX 4.24 -CenterY 1.99 -Color $script:Warm -LightColor "#F2CB9F" -Mode "Vector"

    # Module 2 lower global branch.
    $null = Add-Fig3Rect -Page $page -Name "module2_global_input" -Left 2.64 -Bottom 0.48 -Right 3.08 -Top 1.20 -Fill $script:CoolLight -Line $script:Cool -LineWeightPt 0.9 -RoundingIn 0.05
    $null = Add-Fig3Text -Page $page -Name "module2_global_input_label" -Text "Global`nhierarchical`nsemantic state" -CenterX 2.86 -CenterY 1.04 -Width 0.50 -Height 0.30 -FontSizePt 5.0
    $null = Add-Fig3Svg -Page $page -Name "asset_global_state_module2" -Path ([string]$assetPaths["global_hierarchical_semantic_state"]) -CenterX 2.86 -CenterY 0.70 -MaxWidth 0.34 -MaxHeight 0.28
    $null = Add-Fig3Rect -Page $page -Name "global_value_branch" -Left 3.23 -Bottom 0.48 -Right 3.81 -Top 1.20 -Fill $script:CoolLight -Line $script:Cool -LineWeightPt 0.9 -RoundingIn 0.05
    $null = Add-Fig3Text -Page $page -Name "global_value_branch_label" -Text "Global value`nbranch" -CenterX 3.52 -CenterY 1.07 -Width 0.50 -Height 0.22 -FontSizePt 5.6 -Bold $true
    Add-Fig3FeatureBlocks -Page $page -Prefix "global_encoder" -CenterX 3.52 -CenterY 0.71 -Color $script:Cool -LightColor "#C8D9EB" -Mode "Encoder"
    $null = Add-Fig3Rect -Page $page -Name "global_exploration_value_feature" -Left 3.96 -Bottom 0.48 -Right 4.52 -Top 1.20 -Fill $script:CoolLight -Line $script:Cool -LineWeightPt 0.9 -RoundingIn 0.05
    $null = Add-Fig3Text -Page $page -Name "global_exploration_value_feature_label" -Text "Global`nexploration-`nvalue feature" -CenterX 4.24 -CenterY 1.04 -Width 0.50 -Height 0.30 -FontSizePt 5.0
    Add-Fig3FeatureBlocks -Page $page -Prefix "global_feature_vector" -CenterX 4.24 -CenterY 0.71 -Color $script:Cool -LightColor "#C8D9EB" -Mode "Vector"

    $null = Add-Fig3Line -Page $page -Name "connector_local_state_to_module2" -BeginX 2.34 -BeginY 2.11 -EndX 2.64 -EndY 2.11 -Color $script:Warm -WeightPt 1.25 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_local_input_to_encoder" -BeginX 3.08 -BeginY 2.11 -EndX 3.23 -EndY 2.11 -Color $script:Warm -WeightPt 1.25 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_local_encoder_to_features" -BeginX 3.81 -BeginY 2.11 -EndX 3.96 -EndY 2.11 -Color $script:Warm -WeightPt 1.25 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_global_state_to_module2" -BeginX 2.34 -BeginY 0.84 -EndX 2.64 -EndY 0.84 -Color $script:Cool -WeightPt 1.25 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_global_input_to_encoder" -BeginX 3.08 -BeginY 0.84 -EndX 3.23 -EndY 0.84 -Color $script:Cool -WeightPt 1.25 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_global_encoder_to_features" -BeginX 3.81 -BeginY 0.84 -EndX 3.96 -EndY 0.84 -Color $script:Cool -WeightPt 1.25 -Arrow $true

    # Module 3 action-value estimation and environment-side selection.
    $null = Add-Fig3Rect -Page $page -Name "local_action_advantages" -Left 4.79 -Bottom 1.82 -Right 5.35 -Top 2.37 -Fill $script:WarmLight -Line $script:Warm -LineWeightPt 0.95 -RoundingIn 0.05 -Text "Local action`nadvantages`nA(s,a)" -FontSizePt 5.5 -Bold $true
    $null = Add-Fig3Rect -Page $page -Name "global_state_value" -Left 4.79 -Bottom 0.75 -Right 5.35 -Top 1.30 -Fill $script:CoolLight -Line $script:Cool -LineWeightPt 0.95 -RoundingIn 0.05 -Text "Global state`nvalue V(s)" -FontSizePt 6.1 -Bold $true
    $null = Add-Fig3Line -Page $page -Name "connector_local_features_to_advantages" -BeginX 4.52 -BeginY 2.11 -EndX 4.79 -EndY 2.11 -Color $script:Warm -WeightPt 1.25 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_global_features_to_value" -BeginX 4.52 -BeginY 0.84 -EndX 4.79 -EndY 1.02 -Color $script:Cool -WeightPt 1.25 -Arrow $true

    $null = Add-Fig3Rect -Page $page -Name "semantic_dueling_fusion" -Left 5.49 -Bottom 0.83 -Right 5.99 -Top 2.23 -Fill $script:GreenLight -Line $script:Green -LineWeightPt 1.0 -RoundingIn 0.08
    $null = Add-Fig3Text -Page $page -Name "semantic_dueling_fusion_label" -Text "Semantic`ndueling fusion" -CenterX 5.74 -CenterY 1.93 -Width 0.47 -Height 0.34 -FontSizePt 5.3 -Bold $true
    $warmCircle = $page.DrawOval(5.60, 1.20, 5.80, 1.40)
    Set-VisioShapeName -Shape $warmCircle -Name "fusion_advantage_symbol"
    Set-VisioNodeStyle -Shape $warmCircle -FillColor "#F2CB9F" -LineColor $script:Warm -LineWeightPt 0.85 -FillTransparencyPercent 14
    $coolCircle = $page.DrawOval(5.70, 1.20, 5.90, 1.40)
    Set-VisioShapeName -Shape $coolCircle -Name "fusion_value_symbol"
    Set-VisioNodeStyle -Shape $coolCircle -FillColor "#C8D9EB" -LineColor $script:Cool -LineWeightPt 0.85 -FillTransparencyPercent 14
    $null = Add-Fig3Text -Page $page -Name "fusion_plus" -Text "+" -CenterX 5.75 -CenterY 1.30 -Width 0.12 -Height 0.12 -FontSizePt 7.0 -Bold $true -Color $script:Green
    $null = Add-Fig3Line -Page $page -Name "connector_advantages_to_fusion" -BeginX 5.35 -BeginY 2.10 -EndX 5.49 -EndY 1.82 -Color $script:Warm -WeightPt 1.2 -Arrow $true
    $null = Add-Fig3Line -Page $page -Name "connector_value_to_fusion" -BeginX 5.35 -BeginY 1.02 -EndX 5.49 -EndY 1.18 -Color $script:Cool -WeightPt 1.2 -Arrow $true

    $null = Add-Fig3Text -Page $page -Name "q_values_label" -Text "Q-values of eight`ncandidate actions" -CenterX 6.37 -CenterY 2.32 -Width 0.80 -Height 0.27 -FontSizePt 5.8 -Bold $true
    Add-Fig3QMatrix -Page $page -CenterX 6.37 -CenterY 1.86
    $null = Add-Fig3Line -Page $page -Name "connector_fusion_to_qvalues" -BeginX 5.99 -BeginY 1.53 -EndX 6.10 -EndY 1.86 -Color $script:Green -WeightPt 1.35 -Arrow $true

    $null = Add-Fig3Rect -Page $page -Name "valid_action_mask" -Left 6.00 -Bottom 0.29 -Right 6.74 -Top 1.02 -Fill "#F8FAFB" -Line $script:Neutral -LineWeightPt 0.95 -RoundingIn 0.06
    $null = Add-Fig3Text -Page $page -Name "valid_action_mask_label" -Text "Valid-action mask`n(environment-side)" -CenterX 6.37 -CenterY 0.85 -Width 0.65 -Height 0.24 -FontSizePt 5.6
    Add-Fig3MaskGlyph -Page $page -CenterX 6.37 -CenterY 0.51
    $null = Add-Fig3Line -Page $page -Name "connector_qvalues_to_mask" -BeginX 6.37 -BeginY 1.59 -EndX 6.37 -EndY 1.02 -Color $script:Green -WeightPt 1.2 -Arrow $true

    $null = Add-Fig3Rect -Page $page -Name "selected_action" -Left 6.74 -Bottom 0.43 -Right 7.15 -Top 0.87 -Fill $script:GreenLight -Line $script:Green -LineWeightPt 1.0 -RoundingIn 0.06 -Text "Selected`naction" -FontSizePt 5.0 -Bold $true
    $null = Add-Fig3Line -Page $page -Name "connector_mask_to_selected_action" -BeginX 6.72 -BeginY 0.65 -EndX 6.74 -EndY 0.65 -Color $script:Green -WeightPt 1.2 -Arrow $true

    $document.SaveAs($vsdxPath)

    # Close and reopen the saved package before producing the final exports.
    $document.Close()
    Release-VisioComObject -ComObject $page
    Release-VisioComObject -ComObject $document
    $page = $null
    $document = $null

    $reopenedDocument = $session.Application.Documents.OpenEx($vsdxPath, 194)
    $reopenedPage = $reopenedDocument.Pages.Item(1)
    $null = Export-VisioPng -Page $reopenedPage -Path $pngPath -Dpi $Dpi
    $reopenedPage.Export($svgPath)
    if (-not (Test-Path -LiteralPath $svgPath) -or (Get-Item -LiteralPath $svgPath).Length -le 0) {
        throw "Visio SVG export was not created or is empty: $svgPath"
    }

    $topLevelCount = [int]$reopenedPage.Shapes.Count
    $recursiveCount = Get-RecursiveShapeCount -Shapes $reopenedPage.Shapes
    $connectorCount = 0
    $embeddedAssetCount = 0
    $requiredText = New-Object System.Collections.Generic.List[string]
    for ($index = 1; $index -le [int]$reopenedPage.Shapes.Count; $index++) {
        $shape = $reopenedPage.Shapes.Item($index)
        try {
            $name = [string]$shape.NameU
            if ($name -like "connector_*") {
                $connectorCount++
            }
            if ($name -like "asset_*") {
                $embeddedAssetCount++
            }
            if (-not [string]::IsNullOrWhiteSpace([string]$shape.Text)) {
                $requiredText.Add([string]$shape.Text)
            }
        }
        finally {
            Release-VisioComObject -ComObject $shape
        }
    }
    $joinedText = [string]::Join("`n", $requiredText)
    foreach ($text in @(
        "Dynamic cumulative",
        "Robot position",
        "Interaction history",
        "occupancy–behavior",
        "Global hierarchical",
        "Fixed-dimensional",
        "Semantic",
        "Q-values of eight",
        "environment-side",
        "Selected"
    )) {
        if (-not $joinedText.Contains($text)) {
            throw "Required Figure 3 text fragment was not found after reopen: $text"
        }
    }
    foreach ($forbidden in @(
        "Robot pose",
        "Fully map-size independent",
        "Arbitrary-size lossless representation",
        "Unlimited map size",
        "replay buffer",
        "reward",
        "target network"
    )) {
        if ($joinedText.Contains($forbidden)) {
            throw "Forbidden Figure 3 text was found after reopen: $forbidden"
        }
    }

    $summary = [ordered]@{
        VsdxPath = $vsdxPath
        SvgPath = $svgPath
        PngPath = $pngPath
        PageWidthIn = [Math]::Round([double]$reopenedPage.PageSheet.CellsU("PageWidth").ResultIU, 3)
        PageHeightIn = [Math]::Round([double]$reopenedPage.PageSheet.CellsU("PageHeight").ResultIU, 3)
        NativeTopLevelShapeCount = $topLevelCount
        RecursiveShapeCount = $recursiveCount
        ConnectorCount = $connectorCount
        IndependentlyImportedSvgAssetCount = $embeddedAssetCount
        ReopenValidation = $true
        RequiredTextValidation = $true
        ForbiddenTextValidation = $true
        PngDpi = $Dpi
        PngPixelWidth = [int][Math]::Round(7.20 * $Dpi)
        PngPixelHeight = [int][Math]::Round(3.08 * $Dpi)
        VisioVersion = [string]$session.Application.Version
    }
    Write-Output ("FIG3_BUILD_SUMMARY_JSON=" + ($summary | ConvertTo-Json -Compress))
}
finally {
    Release-VisioComObject -ComObject $reopenedPage
    if ($null -ne $reopenedDocument) {
        try { $reopenedDocument.Close() } catch {}
    }
    Release-VisioComObject -ComObject $reopenedDocument
    Release-VisioComObject -ComObject $page
    if ($null -ne $document) {
        try { $document.Close() } catch {}
    }
    Release-VisioComObject -ComObject $document
    Stop-VisioSession -Session $session
}
