[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AssetsDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputVsdx,
    [Parameter(Mandatory = $true)]
    [string]$OutputPng,
    [Parameter(Mandatory = $true)]
    [string]$OutputPdf,
    [int]$Dpi = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$commonPath = Join-Path $PSScriptRoot "visio_common.ps1"
. $commonPath

$script:FontName = "Arial"
$script:MathFontName = "Cambria Math"
$script:Ink = "#19324A"
$script:ModuleLine = "#7E91A4"
$script:ModuleFill = "#FCFDFE"
$script:Warm = "#C96144"
$script:WarmMid = "#E99D4E"
$script:WarmLight = "#FBF0EA"
$script:Cool = "#5185C0"
$script:CoolDark = "#315F91"
$script:CoolLight = "#EEF4FA"
$script:Green = "#55966B"
$script:GreenLight = "#EEF7F0"
$script:Neutral = "#607487"
$script:NeutralLight = "#F4F7F9"

function Resolve-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [IO.Path]::GetFullPath($Path)
}

function Add-Fig4Rect {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][double]$Left,
        [Parameter(Mandatory = $true)][double]$Bottom,
        [Parameter(Mandatory = $true)][double]$Right,
        [Parameter(Mandatory = $true)][double]$Top,
        [Parameter(Mandatory = $true)][string]$Fill,
        [Parameter(Mandatory = $true)][string]$Line,
        [double]$LineWeightPt = 0.85,
        [double]$RoundingIn = 0.045,
        [string]$Text = "",
        [double]$FontSizePt = 6.0,
        [bool]$Bold = $false,
        [string]$TextColor = "",
        [string]$FontName = ""
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
        Set-VisioTextFormat `
            -Shape $shape `
            -FontName $(if ([string]::IsNullOrWhiteSpace($FontName)) { $script:FontName } else { $FontName }) `
            -FontSizePt $FontSizePt `
            -HorizontalAlign "Center" `
            -VerticalAlign "Middle" `
            -TextColor $(if ([string]::IsNullOrWhiteSpace($TextColor)) { $script:Ink } else { $TextColor })
        if ($Bold) {
            Set-VisioCellFormula -Shape $shape -CellName "Char.Style" -Formula "1"
        }
        foreach ($marginName in @("LeftMargin", "RightMargin", "TopMargin", "BottomMargin")) {
            Set-VisioCellFormula -Shape $shape -CellName $marginName -Formula "0.015 in" -Optional
        }
    }
    return ,$shape
}

function Add-Fig4Text {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][double]$Width,
        [Parameter(Mandatory = $true)][double]$Height,
        [double]$FontSizePt = 6.0,
        [bool]$Bold = $false,
        [string]$Color = "",
        [string]$FontName = "",
        [ValidateSet("Left", "Center", "Right")]
        [string]$HorizontalAlign = "Center"
    )
    $shape = Add-VisioTextBlock `
        -Page $Page `
        -Name $Name `
        -Text $Text `
        -CenterX $CenterX `
        -CenterY $CenterY `
        -Width $Width `
        -Height $Height `
        -FontName $(if ([string]::IsNullOrWhiteSpace($FontName)) { $script:FontName } else { $FontName }) `
        -FontSizePt $FontSizePt `
        -HorizontalAlign $HorizontalAlign `
        -TextColor $(if ([string]::IsNullOrWhiteSpace($Color)) { $script:Ink } else { $Color })
    if ($Bold) {
        Set-VisioCellFormula -Shape $shape -CellName "Char.Style" -Formula "1"
    }
    foreach ($marginName in @("LeftMargin", "RightMargin", "TopMargin", "BottomMargin")) {
        Set-VisioCellFormula -Shape $shape -CellName $marginName -Formula "0.005 in" -Optional
    }
    return ,$shape
}

function Add-Fig4Line {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][double]$BeginX,
        [Parameter(Mandatory = $true)][double]$BeginY,
        [Parameter(Mandatory = $true)][double]$EndX,
        [Parameter(Mandatory = $true)][double]$EndY,
        [Parameter(Mandatory = $true)][string]$Color,
        [double]$WeightPt = 0.9,
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

function Add-Fig4Asset {
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
        throw "Required Figure 4 asset is missing: $Path"
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

function Add-Fig4StepBadge {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY
    )
    $shape = $Page.DrawOval(
        $CenterX - 0.085,
        $CenterY - 0.085,
        $CenterX + 0.085,
        $CenterY + 0.085
    )
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioNodeStyle -Shape $shape -FillColor $script:Cool -LineColor $script:Cool -LineWeightPt 0.5
    $shape.Text = $Text
    Set-VisioTextFormat -Shape $shape -FontName $script:FontName -FontSizePt 6.1 -TextColor "#FFFFFF"
    Set-VisioCellFormula -Shape $shape -CellName "Char.Style" -Formula "1"
    return ,$shape
}

function Add-Fig4HierarchyGlyph {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][int]$EntranceCount
    )
    $shown = [Math]::Min(3, [Math]::Max(1, $EntranceCount))
    $parent = Add-Fig4Rect `
        -Page $Page `
        -Name "hierarchy_parent_b1" `
        -Left ($CenterX - 0.23) `
        -Bottom ($CenterY + 0.09) `
        -Right ($CenterX + 0.23) `
        -Top ($CenterY + 0.34) `
        -Fill "#C8D9EB" `
        -Line $script:CoolDark `
        -LineWeightPt 0.9 `
        -RoundingIn 0.04 `
        -Text "B1" `
        -FontSizePt 6.2 `
        -Bold $true
    $childXs = if ($shown -eq 1) {
        @($CenterX)
    }
    elseif ($shown -eq 2) {
        @(
            ([double]$CenterX - 0.16),
            ([double]$CenterX + 0.16)
        )
    }
    else {
        @(
            ([double]$CenterX - 0.22),
            [double]$CenterX,
            ([double]$CenterX + 0.22)
        )
    }
    for ($index = 0; $index -lt $shown; $index++) {
        $childX = [double]$childXs[$index]
        $null = Add-Fig4Line `
            -Page $Page `
            -Name "hierarchy_edge_$($index + 1)" `
            -BeginX $CenterX `
            -BeginY ($CenterY + 0.09) `
            -EndX $childX `
            -EndY ($CenterY - 0.17) `
            -Color $script:Cool `
            -WeightPt 0.75
        $null = Add-Fig4Rect `
            -Page $Page `
            -Name "hierarchy_entrance_e$($index + 1)" `
            -Left ($childX - 0.085) `
            -Bottom ($CenterY - 0.31) `
            -Right ($childX + 0.085) `
            -Top ($CenterY - 0.17) `
            -Fill $script:CoolLight `
            -Line $script:Cool `
            -LineWeightPt 0.7 `
            -RoundingIn 0.018 `
            -Text "E$($index + 1)" `
            -FontSizePt 4.8
    }
    if ($EntranceCount -gt $shown) {
        $null = Add-Fig4Text `
            -Page $Page `
            -Name "hierarchy_ellipsis" `
            -Text "..." `
            -CenterX ($CenterX + 0.34) `
            -CenterY ($CenterY - 0.23) `
            -Width 0.18 `
            -Height 0.12 `
            -FontSizePt 6.4 `
            -Color $script:CoolDark
    }
    Release-VisioComObject -ComObject $parent
}

function Add-Fig4PackingGlyph {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY
    )
    $specs = @(
        @{ Name = "packing_B"; X = $CenterX - 0.24; Y = $CenterY + 0.15; W = 0.16; H = 0.34; Fill = "#C8D9EB"; Line = $script:Cool; Text = "B" },
        @{ Name = "packing_E"; X = $CenterX + 0.09; Y = $CenterY + 0.15; W = 0.34; H = 0.34; Fill = "#E8F0F8"; Line = $script:Cool; Text = "E" },
        @{ Name = "packing_MB"; X = $CenterX - 0.24; Y = $CenterY - 0.22; W = 0.16; H = 0.23; Fill = "#DCEBDD"; Line = $script:Green; Text = "M_B" },
        @{ Name = "packing_ME"; X = $CenterX + 0.09; Y = $CenterY - 0.22; W = 0.34; H = 0.23; Fill = "#EEF7F0"; Line = $script:Green; Text = "M_E" }
    )
    foreach ($spec in $specs) {
        $null = Add-Fig4Rect `
            -Page $Page `
            -Name ([string]$spec.Name) `
            -Left ([double]$spec.X - ([double]$spec.W / 2.0)) `
            -Bottom ([double]$spec.Y - ([double]$spec.H / 2.0)) `
            -Right ([double]$spec.X + ([double]$spec.W / 2.0)) `
            -Top ([double]$spec.Y + ([double]$spec.H / 2.0)) `
            -Fill ([string]$spec.Fill) `
            -Line ([string]$spec.Line) `
            -LineWeightPt 0.65 `
            -RoundingIn 0.01 `
            -Text ([string]$spec.Text) `
            -FontSizePt 4.2 `
            -FontName $script:MathFontName
    }
}

function Get-RecursiveShapeStats {
    param([Parameter(Mandatory = $true)]$Shapes)
    $count = 0
    $foreign = 0
    $oneDimensional = 0
    for ($index = 1; $index -le [int]$Shapes.Count; $index++) {
        $shape = $Shapes.Item($index)
        try {
            $count++
            if ([int]$shape.Type -eq 4) {
                $foreign++
            }
            try {
                if ([int]$shape.OneD -ne 0) {
                    $oneDimensional++
                }
            }
            catch {}
            if ([int]$shape.Type -eq 2) {
                $childStats = Get-RecursiveShapeStats -Shapes $shape.Shapes
                $count += [int]$childStats.Count
                $foreign += [int]$childStats.Foreign
                $oneDimensional += [int]$childStats.OneDimensional
            }
        }
        finally {
            Release-VisioComObject -ComObject $shape
        }
    }
    return [pscustomobject]@{
        Count = $count
        Foreign = $foreign
        OneDimensional = $oneDimensional
    }
}

$assetNames = @(
    "belief_map_with_robot_and_history",
    "robot_centered_local_window",
    "local_channel_free_space",
    "local_channel_obstacle",
    "local_channel_visit_count",
    "local_channel_recent_trajectory",
    "frontier_unknown_region_extraction"
)
$assetsPath = Resolve-AbsolutePath -Path $AssetsDir
$assetPaths = @{}
foreach ($assetName in $assetNames) {
    # Visio can incorrectly reuse Matplotlib SVG image-resource identifiers
    # across separately imported files. Embed each 300 dpi PNG independently;
    # the exporter still provides parallel SVGs for inspection and reuse.
    $assetPath = Join-Path $assetsPath "${assetName}.png"
    if (-not (Test-Path -LiteralPath $assetPath)) {
        throw "Required Figure 4 asset is missing: $assetPath"
    }
    $assetPaths[$assetName] = $assetPath
}
$manifestPath = Join-Path $assetsPath "fig4_state_construction_assets_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Required Figure 4 asset manifest is missing: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ([int]$manifest.seed -ne 1 -or [int]$manifest.resolved_step -ne 8) {
    throw "Figure 4 requires the verified seed-1/step-8 scene."
}
if (
    [int]$manifest.local_state_shape[0] -ne 4 -or
    [int]$manifest.local_state_shape[1] -ne 21 -or
    [int]$manifest.local_state_shape[2] -ne 21
) {
    throw "Figure 4 local-state manifest is not 4 x 21 x 21."
}
$entranceCount = [int]$manifest.global_semantic_state.frontier_entrance_count

$vsdxPath = Resolve-AbsolutePath -Path $OutputVsdx
$pngPath = Resolve-AbsolutePath -Path $OutputPng
$pdfPath = Resolve-AbsolutePath -Path $OutputPdf
foreach ($path in @($vsdxPath, $pngPath, $pdfPath)) {
    $directory = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $directory)) {
        $null = New-Item -ItemType Directory -Path $directory
    }
    if (Test-VisioFileLocked -Path $path) {
        throw "Output is locked by another application: $path"
    }
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
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
        -PageHeightIn 3.45 `
        -PageName "Fig4_State_Construction"
    $document = $bundle.Document
    $page = $bundle.Page

    # Four restrained manuscript panels; no figure title is embedded.
    $null = Add-Fig4Rect -Page $page -Name "panel_a_frame" -Left 0.03 -Bottom 0.05 -Right 1.46 -Top 3.40 -Fill $script:ModuleFill -Line $script:ModuleLine -LineWeightPt 0.85 -RoundingIn 0.055
    $null = Add-Fig4Rect -Page $page -Name "panel_b_frame" -Left 1.53 -Bottom 0.05 -Right 3.50 -Top 3.40 -Fill $script:WarmLight -Line $script:Warm -LineWeightPt 0.90 -RoundingIn 0.055
    $null = Add-Fig4Rect -Page $page -Name "panel_c_frame" -Left 3.57 -Bottom 0.05 -Right 5.96 -Top 3.40 -Fill $script:CoolLight -Line $script:Cool -LineWeightPt 0.90 -RoundingIn 0.055
    $null = Add-Fig4Rect -Page $page -Name "panel_d_frame" -Left 6.03 -Bottom 0.05 -Right 7.17 -Top 3.40 -Fill $script:NeutralLight -Line $script:Neutral -LineWeightPt 0.90 -RoundingIn 0.055

    $null = Add-Fig4Text -Page $page -Name "panel_a_title" -Text "(a) Dynamic cumulative belief map" -CenterX 0.745 -CenterY 3.20 -Width 1.30 -Height 0.22 -FontSizePt 6.8 -Bold $true
    $null = Add-Fig4Text -Page $page -Name "panel_b_title" -Text "(b) Local occupancy-behavior state" -CenterX 2.515 -CenterY 3.20 -Width 1.82 -Height 0.22 -FontSizePt 6.8 -Bold $true -Color $script:Warm
    $null = Add-Fig4Text -Page $page -Name "panel_c_title" -Text "(c) Global hierarchical semantic state" -CenterX 4.765 -CenterY 3.20 -Width 2.20 -Height 0.22 -FontSizePt 6.8 -Bold $true -Color $script:CoolDark
    $null = Add-Fig4Text -Page $page -Name "panel_d_title" -Text "(d) Fixed-dimensional`nstate interface" -CenterX 6.60 -CenterY 3.16 -Width 1.02 -Height 0.32 -FontSizePt 6.6 -Bold $true

    # Panel (a): one real snapshot with actual step-7/step-8 storage boundaries.
    $null = Add-Fig4Asset -Page $page -Name "asset_belief_map_with_robot_history" -Path ([string]$assetPaths["belief_map_with_robot_and_history"]) -CenterX 0.745 -CenterY 2.33 -MaxWidth 1.24 -MaxHeight 1.10
    $null = Add-Fig4Line -Page $page -Name "legend_previous_boundary_line" -BeginX 0.17 -BeginY 1.62 -EndX 0.43 -EndY 1.62 -Color $script:Neutral -WeightPt 0.85 -Dashed $true
    $null = Add-Fig4Text -Page $page -Name "legend_previous_boundary" -Text "previous boundary" -CenterX 0.88 -CenterY 1.62 -Width 0.78 -Height 0.16 -FontSizePt 5.2 -HorizontalAlign "Left"
    $null = Add-Fig4Line -Page $page -Name "legend_expanded_boundary_line" -BeginX 0.17 -BeginY 1.37 -EndX 0.43 -EndY 1.37 -Color $script:CoolDark -WeightPt 1.05
    $null = Add-Fig4Text -Page $page -Name "legend_expanded_boundary" -Text "expanded boundary" -CenterX 0.88 -CenterY 1.37 -Width 0.78 -Height 0.16 -FontSizePt 5.2 -HorizontalAlign "Left"
    $null = Add-Fig4Line -Page $page -Name "legend_local_window_line" -BeginX 0.17 -BeginY 1.12 -EndX 0.43 -EndY 1.12 -Color $script:Warm -WeightPt 1.05 -Dashed $true
    $null = Add-Fig4Text -Page $page -Name "legend_local_window" -Text "21 x 21 local window" -CenterX 0.88 -CenterY 1.12 -Width 0.78 -Height 0.16 -FontSizePt 5.2 -HorizontalAlign "Left"
    $null = Add-Fig4Text -Page $page -Name "panel_a_note" -Text "Known free space  |  obstacles  |  unknown`nRobot and executed trajectory" -CenterX 0.745 -CenterY 0.55 -Width 1.20 -Height 0.38 -FontSizePt 4.9 -Color $script:Neutral

    # Panel (b): real robot-centered crop and four separate implementation channels.
    $null = Add-Fig4Text -Page $page -Name "local_window_label" -Text "Robot-centered local window" -CenterX 1.94 -CenterY 2.90 -Width 0.75 -Height 0.18 -FontSizePt 5.2 -Bold $true -Color $script:Warm
    $null = Add-Fig4Asset -Page $page -Name "asset_robot_centered_local_window" -Path ([string]$assetPaths["robot_centered_local_window"]) -CenterX 1.94 -CenterY 2.35 -MaxWidth 0.68 -MaxHeight 0.68
    $channelCards = @(
        @{ Key = "local_channel_free_space"; Name = "free"; Label = "Free space"; X = 2.69; Y = 2.50 },
        @{ Key = "local_channel_obstacle"; Name = "obstacle"; Label = "Obstacle"; X = 3.17; Y = 2.50 },
        @{ Key = "local_channel_visit_count"; Name = "visit"; Label = "Visit count"; X = 2.69; Y = 1.72 },
        @{ Key = "local_channel_recent_trajectory"; Name = "trajectory"; Label = "Recent trajectory"; X = 3.17; Y = 1.72 }
    )
    foreach ($card in $channelCards) {
        $null = Add-Fig4Text -Page $page -Name "channel_$($card.Name)_label" -Text ([string]$card.Label) -CenterX ([double]$card.X) -CenterY ([double]$card.Y + 0.34) -Width 0.44 -Height 0.15 -FontSizePt 4.6
        $null = Add-Fig4Asset -Page $page -Name "asset_channel_$($card.Name)" -Path ([string]$assetPaths[[string]$card.Key]) -CenterX ([double]$card.X) -CenterY ([double]$card.Y) -MaxWidth 0.43 -MaxHeight 0.43
    }
    $null = Add-Fig4Line -Page $page -Name "connector_local_crop_to_channels" -BeginX 2.31 -BeginY 2.35 -EndX 2.43 -EndY 2.35 -Color $script:Warm -WeightPt 1.05 -Arrow $true
    $null = Add-Fig4Rect -Page $page -Name "local_tensor_output" -Left 1.84 -Bottom 0.54 -Right 3.29 -Top 1.04 -Fill "#FFFFFF" -Line $script:Warm -LineWeightPt 0.9 -RoundingIn 0.05 -Text "S_t^loc ∈ R^(4×21×21)" -FontSizePt 7.0 -Bold $true -FontName $script:MathFontName
    $null = Add-Fig4Text -Page $page -Name "local_history_note" -Text "older  →  newer (linear weighting)" -CenterX 2.98 -CenterY 1.25 -Width 0.90 -Height 0.16 -FontSizePt 4.5 -Color $script:Warm

    # A visible shared-source split without implying local-to-global causation.
    $junction = $page.DrawOval(1.455, 2.05, 1.505, 2.10)
    Set-VisioShapeName -Shape $junction -Name "shared_snapshot_junction"
    Set-VisioNodeStyle -Shape $junction -FillColor "#FFFFFF" -LineColor $script:Neutral -LineWeightPt 0.75
    $null = Add-Fig4Line -Page $page -Name "connector_map_to_split" -BeginX 1.38 -BeginY 2.075 -EndX 1.48 -EndY 2.075 -Color $script:Neutral -WeightPt 0.9
    $null = Add-Fig4Line -Page $page -Name "connector_split_to_local" -BeginX 1.48 -BeginY 2.075 -EndX 1.63 -EndY 2.35 -Color $script:Warm -WeightPt 1.1 -Arrow $true
    $null = Add-Fig4Text -Page $page -Name "shared_snapshot_label" -Text "same B_t" -CenterX 1.49 -CenterY 1.91 -Width 0.34 -Height 0.12 -FontSizePt 4.1 -Color $script:Neutral

    # Panel (c): three-step global construction plus compact feature definitions.
    $stepCards = @(
        @{ Index = 1; X0 = 3.68; X1 = 4.35; Title = "Frontier-unknown-`nregion extraction" },
        @{ Index = 2; X0 = 4.40; X1 = 5.15; Title = "Unknown-region block-`nfrontier entrance hierarchy" },
        @{ Index = 3; X0 = 5.20; X1 = 5.85; Title = "Feature construction and`nfixed-capacity packing" }
    )
    foreach ($card in $stepCards) {
        $null = Add-Fig4Rect -Page $page -Name "global_step_$($card.Index)_frame" -Left ([double]$card.X0) -Bottom 1.20 -Right ([double]$card.X1) -Top 2.91 -Fill "#FFFFFF" -Line "#9AB9D8" -LineWeightPt 0.7 -RoundingIn 0.035
        $null = Add-Fig4StepBadge -Page $page -Name "global_step_$($card.Index)_badge" -Text ([string]$card.Index) -CenterX ([double]$card.X0 + 0.12) -CenterY 2.77
        $null = Add-Fig4Text -Page $page -Name "global_step_$($card.Index)_title" -Text ([string]$card.Title) -CenterX (([double]$card.X0 + [double]$card.X1) / 2.0 + 0.04) -CenterY 2.67 -Width ([double]$card.X1 - [double]$card.X0 - 0.12) -Height 0.34 -FontSizePt 4.7 -Bold $true
    }
    $null = Add-Fig4Asset -Page $page -Name "asset_frontier_unknown_extraction" -Path ([string]$assetPaths["frontier_unknown_region_extraction"]) -CenterX 4.015 -CenterY 1.95 -MaxWidth 0.57 -MaxHeight 0.62
    $null = Add-Fig4Text -Page $page -Name "frontier_legend" -Text "Unknown region`nFrontier cluster" -CenterX 4.015 -CenterY 1.42 -Width 0.56 -Height 0.24 -FontSizePt 4.2 -Color $script:CoolDark
    Add-Fig4HierarchyGlyph -Page $page -CenterX 4.775 -CenterY 1.92 -EntranceCount $entranceCount
    $null = Add-Fig4Text -Page $page -Name "hierarchy_caption" -Text "parent block`nchild entrances" -CenterX 4.775 -CenterY 1.37 -Width 0.55 -Height 0.22 -FontSizePt 4.2 -Color $script:CoolDark
    Add-Fig4PackingGlyph -Page $page -CenterX 5.55 -CenterY 1.91
    $null = Add-Fig4Text -Page $page -Name "packing_caption" -Text "N_max = 16`nM_max = 8" -CenterX 5.55 -CenterY 1.38 -Width 0.48 -Height 0.22 -FontSizePt 4.4 -Color $script:CoolDark -FontName $script:MathFontName
    $null = Add-Fig4Line -Page $page -Name "connector_global_step_1_to_2" -BeginX 4.35 -BeginY 2.05 -EndX 4.40 -EndY 2.05 -Color $script:Cool -WeightPt 0.9 -Arrow $true
    $null = Add-Fig4Line -Page $page -Name "connector_global_step_2_to_3" -BeginX 5.15 -BeginY 2.05 -EndX 5.20 -EndY 2.05 -Color $script:Cool -WeightPt 0.9 -Arrow $true
    $null = Add-Fig4Rect -Page $page -Name "global_feature_summary" -Left 3.76 -Bottom 0.22 -Right 5.78 -Top 1.02 -Fill "#FFFFFF" -Line $script:Cool -LineWeightPt 0.75 -RoundingIn 0.035
    $null = Add-Fig4Text -Page $page -Name "block_feature_equation" -Text "b_(t,i) = [relative area, number of entrances]" -CenterX 4.77 -CenterY 0.77 -Width 1.86 -Height 0.18 -FontSizePt 5.0 -FontName $script:MathFontName
    $null = Add-Fig4Text -Page $page -Name "entrance_feature_equation" -Text "e_(t,i,j) = [Δr/H, Δc/W, frontier-cluster scale, obstacle density]" -CenterX 4.77 -CenterY 0.49 -Width 1.92 -Height 0.28 -FontSizePt 4.7 -FontName $script:MathFontName

    # Global branch rail from the same snapshot, kept outside Panel (b) content.
    $null = Add-Fig4Line -Page $page -Name "connector_split_global_drop" -BeginX 1.48 -BeginY 2.075 -EndX 1.48 -EndY 0.12 -Color $script:Cool -WeightPt 0.95
    $null = Add-Fig4Line -Page $page -Name "connector_split_global_rail" -BeginX 1.48 -BeginY 0.12 -EndX 3.63 -EndY 0.12 -Color $script:Cool -WeightPt 0.95
    $null = Add-Fig4Line -Page $page -Name "connector_global_rise" -BeginX 3.63 -BeginY 0.12 -EndX 3.63 -EndY 2.05 -Color $script:Cool -WeightPt 0.95
    $null = Add-Fig4Line -Page $page -Name "connector_split_to_global" -BeginX 3.63 -BeginY 2.05 -EndX 3.68 -EndY 2.05 -Color $script:Cool -WeightPt 1.05 -Arrow $true

    # Panel (d): one compact interface object, not five oversized cards.
    $null = Add-Fig4Rect -Page $page -Name "fixed_interface_group" -Left 6.14 -Bottom 0.72 -Right 7.06 -Top 2.88 -Fill "#FFFFFF" -Line $script:Green -LineWeightPt 0.95 -RoundingIn 0.055
    $interfaceRows = @(
        @{ Name = "interface_local"; Text = "S_t^loc ∈ R^(4×21×21)"; Fill = $script:WarmLight; Line = $script:Warm },
        @{ Name = "interface_block"; Text = "B_t ∈ R^(16×2)"; Fill = $script:CoolLight; Line = $script:Cool },
        @{ Name = "interface_entry"; Text = "E_t ∈ R^(16×8×4)"; Fill = $script:CoolLight; Line = $script:Cool },
        @{ Name = "interface_block_mask"; Text = "M_t^B ∈ {0,1}^16"; Fill = $script:GreenLight; Line = $script:Green },
        @{ Name = "interface_entry_mask"; Text = "M_t^E ∈ {0,1}^(16×8)"; Fill = $script:GreenLight; Line = $script:Green }
    )
    for ($index = 0; $index -lt $interfaceRows.Count; $index++) {
        $top = 2.72 - (0.37 * $index)
        $row = $interfaceRows[$index]
        $null = Add-Fig4Rect `
            -Page $page `
            -Name ([string]$row.Name) `
            -Left 6.23 `
            -Bottom ($top - 0.27) `
            -Right 6.97 `
            -Top $top `
            -Fill ([string]$row.Fill) `
            -Line ([string]$row.Line) `
            -LineWeightPt 0.65 `
            -RoundingIn 0.025 `
            -Text ([string]$row.Text) `
            -FontSizePt 5.25 `
            -FontName $script:MathFontName
    }
    $null = Add-Fig4Text -Page $page -Name "interface_key_note" -Text "five policy inputs" -CenterX 6.60 -CenterY 0.88 -Width 0.80 -Height 0.16 -FontSizePt 5.0 -Bold $true -Color $script:Green
    $null = Add-Fig4Text -Page $page -Name "map_size_decoupling_summary" -Text "Local fixed window + global hierarchical semantic abstraction`n→ five fixed-dimensional inputs" -CenterX 6.60 -CenterY 0.36 -Width 0.94 -Height 0.42 -FontSizePt 4.5 -Color $script:Neutral
    $null = Add-Fig4Line -Page $page -Name "connector_global_to_interface" -BeginX 5.96 -BeginY 1.72 -EndX 6.14 -EndY 1.72 -Color $script:Green -WeightPt 1.05 -Arrow $true

    $document.SaveAs($vsdxPath)

    # Reopen the saved OPC package before producing final previews.
    $document.Close()
    Release-VisioComObject -ComObject $page
    Release-VisioComObject -ComObject $document
    $page = $null
    $document = $null

    $reopenedDocument = $session.Application.Documents.OpenEx($vsdxPath, 194)
    $reopenedPage = $reopenedDocument.Pages.Item(1)
    $pngResult = Export-VisioPng -Page $reopenedPage -Path $pngPath -Dpi $Dpi
    # visFixedFormatPDF=1, visDocExIntentPrint=0, visPrintAll=0.
    $reopenedDocument.ExportAsFixedFormat(1, $pdfPath, 0, 0, 1, 1, $false, $true, $true, $true, $false)
    if (-not (Test-Path -LiteralPath $pdfPath) -or (Get-Item -LiteralPath $pdfPath).Length -le 0) {
        throw "Visio PDF export was not created or is empty: $pdfPath"
    }

    $topLevelCount = [int]$reopenedPage.Shapes.Count
    $stats = Get-RecursiveShapeStats -Shapes $reopenedPage.Shapes
    $importedAssetCount = 0
    $connectorCount = 0
    $allText = New-Object System.Collections.Generic.List[string]
    for ($index = 1; $index -le [int]$reopenedPage.Shapes.Count; $index++) {
        $shape = $reopenedPage.Shapes.Item($index)
        try {
            $name = [string]$shape.NameU
            if ($name -like "asset_*") {
                $importedAssetCount++
            }
            if ($name -like "connector_*") {
                $connectorCount++
            }
            if (-not [string]::IsNullOrWhiteSpace([string]$shape.Text)) {
                $allText.Add([string]$shape.Text)
            }
        }
        finally {
            Release-VisioComObject -ComObject $shape
        }
    }
    $joinedText = [string]::Join("`n", $allText)
    foreach ($required in @(
        "Dynamic cumulative belief map",
        "previous boundary",
        "expanded boundary",
        "Local occupancy-behavior state",
        "Free space",
        "Obstacle",
        "Visit count",
        "Recent trajectory",
        "Global hierarchical semantic state",
        "Frontier-unknown-",
        "frontier entrance hierarchy",
        "frontier-cluster scale",
        "Fixed-dimensional",
        "S_t^loc",
        "B_t",
        "E_t",
        "M_t^B",
        "M_t^E"
    )) {
        if (-not $joinedText.Contains($required)) {
            throw "Required Figure 4 text fragment was not found after reopen: $required"
        }
    }
    foreach ($forbidden in @(
        "reachable unknown block",
        "reachable entrance",
        "entry width",
        "corridor width",
        "physical corridor width",
        "truth-map input",
        "fully map-size independent",
        "arbitrary-size lossless representation",
        "valid-action mask",
        "local frontier raster channel",
        "CNN",
        "advantage head",
        "value head",
        "dueling fusion",
        "Q-value",
        "action selection"
    )) {
        if ($joinedText.ToLowerInvariant().Contains($forbidden.ToLowerInvariant())) {
            throw "Forbidden Figure 4 text was found after reopen: $forbidden"
        }
    }
    if ($importedAssetCount -ne 7) {
        throw "Expected seven independently imported SVG asset objects, found $importedAssetCount."
    }
    if ([int]$stats.Count -le $importedAssetCount) {
        throw "Figure 4 appears to contain only imported assets rather than editable native structure."
    }

    $summary = [ordered]@{
        VsdxPath = $vsdxPath
        PngPath = $pngPath
        PdfPath = $pdfPath
        PageWidthIn = [Math]::Round([double]$reopenedPage.PageSheet.CellsU("PageWidth").ResultIU, 3)
        PageHeightIn = [Math]::Round([double]$reopenedPage.PageSheet.CellsU("PageHeight").ResultIU, 3)
        NativeTopLevelShapeCount = $topLevelCount
        RecursiveShapeCount = [int]$stats.Count
        ForeignObjectCount = [int]$stats.Foreign
        OneDimensionalShapeCount = [int]$stats.OneDimensional
        ConnectorCount = $connectorCount
        IndependentlyImportedAssetCount = $importedAssetCount
        ReopenValidation = $true
        RequiredTextValidation = $true
        ForbiddenTextValidation = $true
        PngDpi = $Dpi
        PngPixelWidth = [int]$pngResult.PixelWidth
        PngPixelHeight = [int]$pngResult.PixelHeight
        VisioVersion = [string]$session.Application.Version
    }
    Write-Output ("FIG4_BUILD_SUMMARY_JSON=" + ($summary | ConvertTo-Json -Compress))
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
