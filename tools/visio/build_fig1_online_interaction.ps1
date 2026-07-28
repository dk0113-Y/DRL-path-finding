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

function Write-VisioStage {
    param([Parameter(Mandatory = $true)][string]$Stage)
    Write-Host "visio_stage=$Stage"
}

function Resolve-Fig1Blueprint {
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
    ) ("figure_demo_blueprint_seed{0}_step{1}_fig1_{2}.json" -f $SeedValue, $StepValue, $PID)
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

function New-Fig1Rectangle {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)]$Layer,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][double]$Width,
        [Parameter(Mandatory = $true)][double]$Height,
        [Parameter(Mandatory = $true)][string]$FillColor,
        [string]$LineColor = "#5D7183",
        [double]$LineWeightPt = 0.65,
        [double]$RoundingIn = 0.025,
        [double]$FillTransparencyPercent = 0.0,
        [string]$Text = "",
        [string]$FontName = "Cambria Math",
        [double]$FontSizePt = 6.8,
        [string]$TextColor = "#233746"
    )
    $shape = $Page.DrawRectangle(
        $CenterX - ($Width / 2.0),
        $CenterY - ($Height / 2.0),
        $CenterX + ($Width / 2.0),
        $CenterY + ($Height / 2.0)
    )
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioNodeStyle `
        -Shape $shape `
        -FillColor $FillColor `
        -LineColor $LineColor `
        -LineWeightPt $LineWeightPt `
        -RoundingIn $RoundingIn `
        -FillTransparencyPercent $FillTransparencyPercent
    if (-not [string]::IsNullOrWhiteSpace($Text)) {
        $shape.Text = $Text
        Set-VisioTextFormat `
            -Shape $shape `
            -FontName $FontName `
            -FontSizePt $FontSizePt `
            -TextColor $TextColor
    }
    Add-ShapeToVisioLayer -Layer $Layer -Shape $shape
    return ,$shape
}

function New-Fig1Oval {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)]$Layer,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][double]$Width,
        [Parameter(Mandatory = $true)][double]$Height,
        [Parameter(Mandatory = $true)][string]$FillColor,
        [string]$LineColor = "#5D7183",
        [double]$LineWeightPt = 0.65
    )
    $shape = $Page.DrawOval(
        $CenterX - ($Width / 2.0),
        $CenterY - ($Height / 2.0),
        $CenterX + ($Width / 2.0),
        $CenterY + ($Height / 2.0)
    )
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioNodeStyle `
        -Shape $shape `
        -FillColor $FillColor `
        -LineColor $LineColor `
        -LineWeightPt $LineWeightPt
    Add-ShapeToVisioLayer -Layer $Layer -Shape $shape
    return ,$shape
}

function Add-Fig1Text {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)]$Layer,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [Parameter(Mandatory = $true)][double]$Width,
        [Parameter(Mandatory = $true)][double]$Height,
        [string]$FontName = "Cambria Math",
        [double]$FontSizePt = 6.8,
        [string]$TextColor = "#233746"
    )
    $shape = Add-VisioTextBlock `
        -Page $Page `
        -Name $Name `
        -Text $Text `
        -CenterX $CenterX `
        -CenterY $CenterY `
        -Width $Width `
        -Height $Height `
        -FontName $FontName `
        -FontSizePt $FontSizePt `
        -TextColor $TextColor
    Add-ShapeToVisioLayer -Layer $Layer -Shape $shape
    return ,$shape
}

function Add-Fig1GridFromMatrix {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)]$Layer,
        [Parameter(Mandatory = $true)][string]$NamePrefix,
        [Parameter(Mandatory = $true)][double]$Left,
        [Parameter(Mandatory = $true)][double]$Bottom,
        [Parameter(Mandatory = $true)][double]$CellSize,
        [Parameter(Mandatory = $true)]$Matrix,
        [Parameter(Mandatory = $true)]$Style,
        [switch]$SkipUnknownCells
    )
    $rowCount = [int]$Matrix.Count
    $colCount = [int]$Matrix[0].Count
    $shapes = @()
    $shapes += New-GridCell `
        -Page $Page `
        -Layer $Layer `
        -Name "${NamePrefix}_background" `
        -Left $Left `
        -Bottom $Bottom `
        -Right ($Left + ($colCount * $CellSize)) `
        -Top ($Bottom + ($rowCount * $CellSize)) `
        -FillColor ([string]$Style.OccupancyPalette.unknown)
    for ($row = 0; $row -lt $rowCount; $row++) {
        if ([int]$Matrix[$row].Count -ne $colCount) {
            throw "Grid matrix rows must have equal lengths."
        }
        for ($col = 0; $col -lt $colCount; $col++) {
            $state = [int]$Matrix[$row][$col]
            if ($SkipUnknownCells -and $state -eq -1) {
                continue
            }
            $fillColor = switch ($state) {
                -1 { [string]$Style.OccupancyPalette.unknown }
                0 { [string]$Style.OccupancyPalette.free }
                1 { [string]$Style.OccupancyPalette.obstacle }
                default { throw "Unknown occupancy state '$state'." }
            }
            $cellLeft = $Left + ($col * $CellSize)
            $cellBottom = $Bottom + (($rowCount - 1 - $row) * $CellSize)
            $shapes += New-GridCell `
                -Page $Page `
                -Layer $Layer `
                -Name ("{0}_cell_r{1:D2}_c{2:D2}" -f $NamePrefix, $row, $col) `
                -Left $cellLeft `
                -Bottom $cellBottom `
                -Right ($cellLeft + $CellSize) `
                -Top ($cellBottom + $CellSize) `
                -FillColor $fillColor
        }
    }
    for ($index = 0; $index -le $colCount; $index++) {
        $x = $Left + ($index * $CellSize)
        $shapes += New-GridLine `
            -Page $Page `
            -Layer $Layer `
            -Name ("{0}_v{1:D2}" -f $NamePrefix, $index) `
            -BeginX $x `
            -BeginY $Bottom `
            -EndX $x `
            -EndY ($Bottom + ($rowCount * $CellSize)) `
            -LineColor ([string]$Style.RadarPalette.grid_line) `
            -LineWeightPt 0.18
    }
    for ($index = 0; $index -le $rowCount; $index++) {
        $y = $Bottom + ($index * $CellSize)
        $shapes += New-GridLine `
            -Page $Page `
            -Layer $Layer `
            -Name ("{0}_h{1:D2}" -f $NamePrefix, $index) `
            -BeginX $Left `
            -BeginY $y `
            -EndX ($Left + ($colCount * $CellSize)) `
            -EndY $y `
            -LineColor ([string]$Style.RadarPalette.grid_line) `
            -LineWeightPt 0.18
    }
    return $shapes
}

function Get-Fig1MatrixCellCenter {
    param(
        [Parameter(Mandatory = $true)][double]$Left,
        [Parameter(Mandatory = $true)][double]$Bottom,
        [Parameter(Mandatory = $true)][double]$CellSize,
        [Parameter(Mandatory = $true)][int]$RowCount,
        [Parameter(Mandatory = $true)][int]$Row,
        [Parameter(Mandatory = $true)][int]$Col
    )
    return [pscustomobject]@{
        X = $Left + (($Col + 0.5) * $CellSize)
        Y = $Bottom + (($RowCount - $Row - 0.5) * $CellSize)
    }
}

function Complete-Fig1IllustrationGroup {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)]$Layer,
        [Parameter(Mandatory = $true)][object[]]$Members,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $group = New-VisioShapeGroup -Page $Page -Shapes $Members -Name $Name -Layer $Layer
    foreach ($member in $Members) {
        Release-VisioComObject -ComObject $member
    }
    return ,$group
}

$outputDirectory = "C:\Users\Dk\Desktop\SCI\paper_picture\visio_outputs"
$requestedOutputPath = Join-Path $outputDirectory "fig1_online_decision_interaction.vsdx"
$requestedPngPath = Join-Path $outputDirectory "fig1_online_decision_interaction.png"
$resolvedBlueprintPath = Resolve-Fig1Blueprint `
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
if ([int]$blueprint.seed -ne 1) {
    throw "Formal Figure 1 assets require seed=1."
}
if ([int]$blueprint.scan_radius -ne 10) {
    throw "Formal Figure 1 assets require scan_radius=10."
}

$nodeNames = @(
    "module_1_local_observation",
    "module_2_belief_update",
    "module_3_policy_state",
    "module_4_online_q_network",
    "module_5_mask_action",
    "module_6_environment_interaction",
    "decision_terminal_condition",
    "terminator_episode_end"
)
$illustrationNames = @(
    "fig1_illustration_local_observation",
    "fig1_illustration_belief_update",
    "fig1_illustration_policy_state",
    "fig1_illustration_online_q",
    "fig1_action_arrows_group",
    "fig1_illustration_environment"
)
$requiredTexts = @(
    "局部传感观测", "oₜ",
    "累计信念地图更新", "Bₜ",
    "策略状态构造", "sₜ",
    "Double DQN在线Q网络", "Q(sₜ,a;θ)",
    "合法动作掩码与动作选择", "aₜ",
    "环境执行与交互反馈", "rₜ, oₜ₊₁, dₜ",
    "满足终止条件？", "否", "是", "回合结束"
)
$forbiddenTexts = @("经验回放", "目标网络", "loss", "参数更新", "n步回报", "Bₜ₋₁")

if ($ValidateOnly) {
    if ([string]::IsNullOrWhiteSpace($ValidationPath)) {
        throw "-ValidationPath is required with -ValidateOnly."
    }
    $validationSession = $null
    $validationDocument = $null
    try {
        $validationSession = Start-VisioSession -Visible $false
        $validationDocument = $validationSession.Application.Documents.OpenEx($ValidationPath, 194)
        $validationResult = Test-VisioOnlineInteractionDocument `
            -Document $validationDocument `
            -RequiredNodeNames $nodeNames `
            -RequiredTexts $requiredTexts `
            -FirstModuleName $nodeNames[0] `
            -DecisionName $nodeNames[6] `
            -TerminatorName $nodeNames[7] `
            -ExpectedMainModuleCount 6 `
            -ExpectedMainConnectorCount 6 `
            -RequiredIllustrationNames $illustrationNames `
            -ForbiddenTexts $forbiddenTexts `
            -ActionArrowGroupName "fig1_action_arrows_group" `
            -Blueprint $blueprint `
            -Style $style
        $validationDocument.Close()
        Release-VisioComObject -ComObject $validationDocument
        $validationDocument = $null
        $validationResult | ConvertTo-Json -Depth 6
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

$session = $null
$workspace = $null
$document = $null
$page = $null
$modules = @()
$decision = $null
$terminator = $null
$illustrationGroups = @()
$layersByName = @{}
$pngResult = $null
$outputPair = $null

try {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    $outputPair = Resolve-VisioOutputPair -VsdxPath $requestedOutputPath -PngPath $requestedPngPath
    $outputPath = [string]$outputPair.VsdxPath
    $pngPath = [string]$outputPair.PngPath
    $session = Start-VisioSession -Visible $false
    Write-VisioStage -Stage "session_started"
    $workspace = New-VisioDocumentPage `
        -Session $session `
        -PageWidthIn 14.60 `
        -PageHeightIn 5.60 `
        -PageName "Fig1_Online_Interaction"
    $document = $workspace.Document
    $page = $workspace.Page
    $layersByName["Illustrations"] = New-VisioLayer -Page $page -Name "Module_Illustrations"

    $centerY = 3.70
    $moduleHeight = 2.75
    $blueFill = "#EDF3F8"
    $greenFill = "#EFF4EC"
    $orangeFill = "#F8EFE6"
    $modules = @(
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[0] -Title "局部传感观测" -Formula "oₜ" -CenterX 1.00 -CenterY $centerY -Width 1.65 -Height $moduleHeight -FillColor $blueFill -TitleFontSizePt 8.8),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[1] -Title "累计信念地图更新" -Formula "Bₜ" -CenterX 2.90 -CenterY $centerY -Width 1.80 -Height $moduleHeight -FillColor $blueFill -TitleFontSizePt 8.4),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[2] -Title "策略状态构造" -Formula "sₜ" -CenterX 4.80 -CenterY $centerY -Width 1.70 -Height $moduleHeight -FillColor $blueFill -TitleFontSizePt 8.8),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[3] -Title "Double DQN在线Q网络" -Formula "Q(sₜ,a;θ)" -CenterX 6.85 -CenterY $centerY -Width 1.95 -Height $moduleHeight -FillColor $greenFill -TitleFontSizePt 8.2 -FormulaFontSizePt 10.2),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[4] -Title "合法动作掩码与动作选择" -Formula "aₜ" -CenterX 9.15 -CenterY $centerY -Width 2.15 -Height $moduleHeight -FillColor $greenFill -TitleFontSizePt 8.1),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[5] -Title "环境执行与交互反馈" -Formula "rₜ, oₜ₊₁, dₜ" -CenterX 11.55 -CenterY $centerY -Width 2.25 -Height $moduleHeight -FillColor $orangeFill -TitleFontSizePt 8.4 -FormulaFontSizePt 9.5)
    )
    $decision = Add-VisioDiamond `
        -Page $page -Name $nodeNames[6] -Text "满足终止条件？" `
        -CenterX 13.65 -CenterY $centerY -Width 1.35 -Height 1.38 -FillColor $orangeFill
    $terminator = Add-VisioTerminator `
        -Page $page -Name $nodeNames[7] -Text "回合结束" `
        -CenterX 13.55 -CenterY 1.02 -Width 1.55 -Height 0.58 -FillColor "#EEE7DD"
    Write-VisioStage -Stage "nodes_created"

    # 1. Real seed-1 local_snap at p_t; no radar rays or action arrows.
    $localCell = 0.050
    $localLeft = 1.00 - (([int]$blueprint.local_shape[1] * $localCell) / 2.0)
    $localBottom = 2.48
    $localCenter = Get-Fig1MatrixCellCenter `
        -Left $localLeft -Bottom $localBottom -CellSize $localCell `
        -RowCount ([int]$blueprint.local_shape[0]) `
        -Row ([int]$blueprint.center_state[0]) `
        -Col ([int]$blueprint.center_state[1])
    $members = @()
    $members += Add-Fig1GridFromMatrix `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_local_obs_grid" `
        -Left $localLeft -Bottom $localBottom -CellSize $localCell `
        -Matrix $blueprint.local_snap_t -Style $style -SkipUnknownCells
    $rangeRadius = [int]$blueprint.scan_radius * $localCell
    $rangeBoundary = $page.DrawOval(
        $localCenter.X - $rangeRadius,
        $localCenter.Y - $rangeRadius,
        $localCenter.X + $rangeRadius,
        $localCenter.Y + $rangeRadius
    )
    Set-VisioShapeName -Shape $rangeBoundary -Name "fig1_local_observation_range"
    Set-VisioCellFormula -Shape $rangeBoundary -CellName "FillPattern" -Formula "0"
    Set-VisioLineStyle `
        -Shape $rangeBoundary `
        -LineColor ([string]$style.RadarPalette.nominal_boundary) `
        -LineWeightPt 0.50 `
        -TransparencyPercent 70.0
    Set-VisioCellFormula -Shape $rangeBoundary -CellName "LinePattern" -Formula "2"
    Add-ShapeToVisioLayer -Layer $layersByName["Illustrations"] -Shape $rangeBoundary
    $members += $rangeBoundary
    $members += Add-PaperFigureRobotParts `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_local_robot" `
        -CenterX $localCenter.X -CenterY $localCenter.Y `
        -CellSizeIn $localCell -Style $style `
        -HeadingAction ([int]$blueprint.selected_action)
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Members $members -Name $illustrationNames[0]

    # 2. A single editable B_t map with one robot at p_t.
    $beliefRows = [int]$blueprint.belief_t.shape[0]
    $beliefCols = [int]$blueprint.belief_t.shape[1]
    $beliefCell = 0.030
    $beliefLeft = 2.90 - (($beliefCols * $beliefCell) / 2.0)
    $beliefBottom = 2.51
    $pTRow = [int]$blueprint.p_t[0] - [int]$blueprint.belief_t.origin_world_rc[0]
    $pTCol = [int]$blueprint.p_t[1] - [int]$blueprint.belief_t.origin_world_rc[1]
    $beliefRobotCenter = Get-Fig1MatrixCellCenter `
        -Left $beliefLeft -Bottom $beliefBottom -CellSize $beliefCell `
        -RowCount $beliefRows -Row $pTRow -Col $pTCol
    $members = @()
    $members += Add-Fig1GridFromMatrix `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_belief_grid" `
        -Left $beliefLeft -Bottom $beliefBottom -CellSize $beliefCell `
        -Matrix $blueprint.belief_t.matrix -Style $style -SkipUnknownCells
    $members += Add-PaperFigureRobotParts `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_belief_robot" `
        -CenterX $beliefRobotCenter.X -CenterY $beliefRobotCenter.Y `
        -CellSizeIn $beliefCell -Style $style `
        -HeadingAction ([int]$blueprint.selected_action)
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Members $members -Name $illustrationNames[1]

    # 3. Policy-state construction: map, pose, and history converge to encoded planes.
    $members = @()
    $members += New-Fig1Rectangle -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_state_map_input" -CenterX 4.18 -CenterY 3.43 -Width 0.34 -Height 0.25 -FillColor "#DCECF7" -LineColor "#5185C0"
    $members += New-Fig1Oval -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_state_position_input" -CenterX 4.18 -CenterY 3.06 -Width 0.16 -Height 0.16 -FillColor "#C96144" -LineColor "#9A4634" -LineWeightPt 0.50
    foreach ($offset in @(-0.12, 0.0, 0.12)) {
        $members += New-Fig1Oval `
            -Page $page -Layer $layersByName["Illustrations"] `
            -Name ("fig1_state_history_{0}" -f ([Math]::Round(($offset + 0.12) * 100))) `
            -CenterX (4.18 + $offset) -CenterY (2.70 + (($offset + 0.12) * 0.40)) `
            -Width 0.085 -Height 0.085 -FillColor "#8EA9D4" -LineColor "#5185C0" -LineWeightPt 0.35
    }
    foreach ($targetY in @(3.43, 3.06, 2.75)) {
        $members += New-ArrowShape `
            -Page $page -Layer $layersByName["Illustrations"] `
            -Name ("fig1_state_merge_{0}" -f ([Math]::Round($targetY * 100))) `
            -BeginX 4.39 -BeginY $targetY -EndX 4.67 -EndY 3.07 `
            -LineColor "#99AABB" -LineWeightPt 0.70 -EndArrow 13 -EndArrowSize 1
    }
    $stackColors = @("#DCECF7", "#C0BEDC", "#99C290")
    for ($index = 0; $index -lt 3; $index++) {
        $members += New-Fig1Rectangle `
            -Page $page -Layer $layersByName["Illustrations"] `
            -Name ("fig1_state_feature_plane_{0}" -f ($index + 1)) `
            -CenterX (4.92 + ($index * 0.07)) -CenterY (2.94 + ($index * 0.09)) `
            -Width 0.72 -Height 0.48 -FillColor $stackColors[$index] `
            -LineColor "#5D7183" -RoundingIn 0.035 -FillTransparencyPercent 12.0
    }
    $members += Add-Fig1Text `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Name "fig1_state_stack_label" -Text "encoded state" `
        -CenterX 5.08 -CenterY 3.46 -Width 0.80 -Height 0.18 `
        -FontName "Arial" -FontSizePt 5.8 -TextColor "#5D7183"
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Members $members -Name $illustrationNames[2]

    # 4. Online Q network only; no replay, target network, or training update.
    $members = @()
    $members += New-Fig1Rectangle `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Name "fig1_q_state_input" -CenterX 6.10 -CenterY 3.08 `
        -Width 0.34 -Height 0.52 -FillColor "#DCECF7" -LineColor "#5185C0" `
        -Text "sₜ" -FontSizePt 7.2
    foreach ($column in 0..1) {
        foreach ($row in 0..2) {
            $members += New-Fig1Oval `
                -Page $page -Layer $layersByName["Illustrations"] `
                -Name ("fig1_q_hidden_c{0}_r{1}" -f ($column + 1), ($row + 1)) `
                -CenterX (6.54 + ($column * 0.37)) -CenterY (2.74 + ($row * 0.34)) `
                -Width 0.15 -Height 0.15 `
                -FillColor $(if ($column -eq 0) { "#8EA9D4" } else { "#99C290" }) `
                -LineColor "#5D7183" -LineWeightPt 0.45
        }
    }
    foreach ($row in 0..2) {
        $members += New-LineShape -Page $page -Layer $layersByName["Illustrations"] -Name ("fig1_q_link_input_{0}" -f ($row + 1)) -BeginX 6.28 -BeginY 3.08 -EndX 6.46 -EndY (2.74 + ($row * 0.34)) -LineColor "#99AABB" -LineWeightPt 0.45 -TransparencyPercent 20.0
        $members += New-LineShape -Page $page -Layer $layersByName["Illustrations"] -Name ("fig1_q_link_hidden_{0}" -f ($row + 1)) -BeginX 6.62 -BeginY (2.74 + ($row * 0.34)) -EndX 6.83 -EndY (2.74 + ($row * 0.34)) -LineColor "#99AABB" -LineWeightPt 0.45 -TransparencyPercent 20.0
    }
    for ($index = 0; $index -lt 8; $index++) {
        $members += New-Fig1Rectangle `
            -Page $page -Layer $layersByName["Illustrations"] `
            -Name ("fig1_q_output_{0:D2}" -f ($index + 1)) `
            -CenterX (7.30 + (($index % 2) * 0.18)) `
            -CenterY (2.68 + ([Math]::Floor($index / 2) * 0.25)) `
            -Width 0.13 -Height 0.13 `
            -FillColor $(if ($index -eq [int]$blueprint.selected_action) { "#C96144" } else { "#F2CB9F" }) `
            -LineColor "#8D6E58" -LineWeightPt 0.35 -RoundingIn 0.012
    }
    $members += New-ArrowShape `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Name "fig1_q_output_link" -BeginX 7.00 -BeginY 3.08 -EndX 7.20 -EndY 3.08 `
        -LineColor "#5D7183" -LineWeightPt 0.85 -EndArrow 13 -EndArrowSize 1
    $members += Add-Fig1Text `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Name "fig1_q_output_label" -Text "Q(a)" `
        -CenterX 7.40 -CenterY 3.72 -Width 0.42 -Height 0.17 -FontSizePt 6.2
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Members $members -Name $illustrationNames[3]

    # 5. Real legal mask: legal green, illegal red, selected thick blue.
    $actionCropRadius = 3
    $actionMatrix = @()
    for (
        $sourceRow = [int]$blueprint.center_state[0] - $actionCropRadius;
        $sourceRow -le [int]$blueprint.center_state[0] + $actionCropRadius;
        $sourceRow++
    ) {
        $actionRow = @()
        for (
            $sourceCol = [int]$blueprint.center_state[1] - $actionCropRadius;
            $sourceCol -le [int]$blueprint.center_state[1] + $actionCropRadius;
            $sourceCol++
        ) {
            $actionRow += [int]$blueprint.local_snap_t[$sourceRow][$sourceCol]
        }
        $actionMatrix += ,@($actionRow)
    }
    $actionCell = 0.140
    $actionLeft = 9.15 - (($actionMatrix[0].Count * $actionCell) / 2.0)
    $actionBottom = 2.50
    $actionCenter = Get-Fig1MatrixCellCenter `
        -Left $actionLeft -Bottom $actionBottom -CellSize $actionCell `
        -RowCount ([int]$actionMatrix.Count) `
        -Row $actionCropRadius `
        -Col $actionCropRadius
    $actionContextMembers = @()
    $actionContextMembers += Add-Fig1GridFromMatrix `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_action_grid" `
        -Left $actionLeft -Bottom $actionBottom -CellSize $actionCell `
        -Matrix $actionMatrix -Style $style -SkipUnknownCells
    $actionContextMembers += Add-PaperFigureRobotParts `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_action_robot" `
        -CenterX $actionCenter.X -CenterY $actionCenter.Y `
        -CellSizeIn $actionCell -Style $style `
        -HeadingAction ([int]$blueprint.selected_action)
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Members $actionContextMembers -Name "fig1_action_context_group"

    $directions = @(
        [pscustomobject]@{ Name = "N"; Index = 0; X = 0.0; Y = 1.0 },
        [pscustomobject]@{ Name = "NE"; Index = 1; X = 1.0; Y = 1.0 },
        [pscustomobject]@{ Name = "E"; Index = 2; X = 1.0; Y = 0.0 },
        [pscustomobject]@{ Name = "SE"; Index = 3; X = 1.0; Y = -1.0 },
        [pscustomobject]@{ Name = "S"; Index = 4; X = 0.0; Y = -1.0 },
        [pscustomobject]@{ Name = "SW"; Index = 5; X = -1.0; Y = -1.0 },
        [pscustomobject]@{ Name = "W"; Index = 6; X = -1.0; Y = 0.0 },
        [pscustomobject]@{ Name = "NW"; Index = 7; X = -1.0; Y = 1.0 }
    )
    $validIndices = @($blueprint.valid_action_indices | ForEach-Object { [int]$_ })
    $actionArrows = @()
    $startRadiusIn = 0.48 * $actionCell
    $actionLengthIn = 1.50 * $actionCell
    foreach ($direction in $directions) {
        $norm = [Math]::Sqrt(
            ([double]$direction.X * [double]$direction.X) +
            ([double]$direction.Y * [double]$direction.Y)
        )
        $unitX = [double]$direction.X / $norm
        $unitY = [double]$direction.Y / $norm
        $state = if ([int]$direction.Index -eq [int]$blueprint.selected_action) {
            "selected"
        }
        elseif ([int]$direction.Index -in $validIndices) {
            "legal"
        }
        else {
            "illegal"
        }
        $beginX = $actionCenter.X + ($unitX * $startRadiusIn)
        $beginY = $actionCenter.Y + ($unitY * $startRadiusIn)
        $actionArrows += New-ArrowShape `
            -Page $page -Layer $layersByName["Illustrations"] `
            -Name ("fig1_action_arrow_{0}" -f ([string]$direction.Name)) `
            -BeginX $beginX -BeginY $beginY `
            -EndX ($beginX + ($unitX * $actionLengthIn)) `
            -EndY ($beginY + ($unitY * $actionLengthIn)) `
            -LineColor ([string]$style.Fig1ActionPalette.$state) `
            -LineWeightPt $(if ($state -eq "selected") {
                [double]$style.Rendering.selected_action_linewidth_pt
            } else {
                [double]$style.Rendering.normal_action_linewidth_pt
            }) `
            -EndArrow 13 `
            -EndArrowSize 1
    }
    $actionArrowGroup = New-VisioShapeGroup `
        -Page $page -Shapes $actionArrows `
        -Name $illustrationNames[4] -Layer $layersByName["Illustrations"]
    foreach ($shape in $actionArrows) { Release-VisioComObject -ComObject $shape }
    $illustrationGroups += $actionArrowGroup
    $actionArrowGroup.BringToFront()

    # 6. The unchanged B_t background with p_t translucent and p_(t+1) solid.
    $environmentLeft = 10.63
    $environmentBottom = $beliefBottom
    $pNextRow = [int]$blueprint.p_t_plus_1[0] - [int]$blueprint.belief_t.origin_world_rc[0]
    $pNextCol = [int]$blueprint.p_t_plus_1[1] - [int]$blueprint.belief_t.origin_world_rc[1]
    $environmentBeforeCenter = Get-Fig1MatrixCellCenter `
        -Left $environmentLeft -Bottom $environmentBottom -CellSize $beliefCell `
        -RowCount $beliefRows -Row $pTRow -Col $pTCol
    $environmentAfterCenter = Get-Fig1MatrixCellCenter `
        -Left $environmentLeft -Bottom $environmentBottom -CellSize $beliefCell `
        -RowCount $beliefRows -Row $pNextRow -Col $pNextCol
    $members = @()
    $members += Add-Fig1GridFromMatrix `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_environment_grid" `
        -Left $environmentLeft -Bottom $environmentBottom -CellSize $beliefCell `
        -Matrix $blueprint.belief_t.matrix -Style $style -SkipUnknownCells
    $members += Add-PaperFigureRobotParts `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_environment_robot_old" `
        -CenterX $environmentBeforeCenter.X -CenterY $environmentBeforeCenter.Y `
        -CellSizeIn $beliefCell -Style $style `
        -HeadingAction ([int]$blueprint.selected_action) `
        -TransparencyPercent 62.0
    $members += Add-PaperFigureRobotParts `
        -Page $page -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_environment_robot_new" `
        -CenterX $environmentAfterCenter.X -CenterY $environmentAfterCenter.Y `
        -CellSizeIn $beliefCell -Style $style `
        -HeadingAction ([int]$blueprint.selected_action)
    $feedback = @(
        [pscustomobject]@{ Text = "rₜ"; Y = 3.43; Fill = "#F2CB9F" },
        [pscustomobject]@{ Text = "oₜ₊₁"; Y = 3.10; Fill = "#DCECF7" },
        [pscustomobject]@{ Text = "dₜ"; Y = 2.77; Fill = "#C0BEDC" }
    )
    for ($index = 0; $index -lt $feedback.Count; $index++) {
        $members += New-Fig1Rectangle `
            -Page $page -Layer $layersByName["Illustrations"] `
            -Name ("fig1_environment_feedback_{0:D2}" -f ($index + 1)) `
            -CenterX 12.24 -CenterY ([double]$feedback[$index].Y) `
            -Width 0.45 -Height 0.24 `
            -FillColor ([string]$feedback[$index].Fill) `
            -LineColor "#8A98A5" -LineWeightPt 0.45 -RoundingIn 0.055 `
            -Text ([string]$feedback[$index].Text) -FontSizePt 6.6
    }
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page -Layer $layersByName["Illustrations"] `
        -Members $members -Name $illustrationNames[5]
    Write-VisioStage -Stage "native_illustrations_created"

    for ($index = 0; $index -lt 5; $index++) {
        $null = Add-VisioDynamicConnector `
            -Page $page -Application $session.Application `
            -FromShape $modules[$index] -ToShape $modules[$index + 1] `
            -Name "connector_main_$($index + 1)"
    }
    $null = Add-VisioDynamicConnector `
        -Page $page -Application $session.Application `
        -FromShape $modules[5] -ToShape $decision -Name "connector_main_6"
    $null = Add-VisioDynamicConnector `
        -Page $page -Application $session.Application `
        -FromShape $decision -ToShape $modules[0] -Name "connector_no_loop" `
        -FromXPercent 0.50 -FromYPercent 1.00 -ToXPercent 0.50 -ToYPercent 1.00 `
        -Dashed -LineWeightPt 1.15
    $null = Add-VisioDynamicConnector `
        -Page $page -Application $session.Application `
        -FromShape $decision -ToShape $terminator -Name "connector_yes_end" `
        -FromXPercent 0.50 -FromYPercent 0.00 -ToXPercent 0.50 -ToYPercent 1.00
    $branchLabel = Add-VisioTextBlock `
        -Page $page -Name "branch_label_no" -Text "否" `
        -CenterX 12.95 -CenterY 5.38 -Width 0.28 -Height 0.20 `
        -FontName "Microsoft YaHei" -FontSizePt 8.8
    Release-VisioComObject -ComObject $branchLabel
    $branchLabel = Add-VisioTextBlock `
        -Page $page -Name "branch_label_yes" -Text "是" `
        -CenterX 13.34 -CenterY 2.18 -Width 0.28 -Height 0.20 `
        -FontName "Microsoft YaHei" -FontSizePt 8.8
    Release-VisioComObject -ComObject $branchLabel

    $null = $document.SaveAs($outputPath)
    $pngResult = Export-VisioPng -Page $page -Path $pngPath -Dpi 300
    Write-VisioStage -Stage "document_saved_and_png_exported"
    $document.Close()

    foreach ($shape in $modules) { Release-VisioComObject -ComObject $shape }
    $modules = @()
    foreach ($shape in $illustrationGroups) { Release-VisioComObject -ComObject $shape }
    $illustrationGroups = @()
    Release-VisioComObject -ComObject $decision
    Release-VisioComObject -ComObject $terminator
    $decision = $null
    $terminator = $null
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
        Validation = $validation
    } | ConvertTo-Json -Depth 10
}
catch {
    throw "Failed to build or validate Figure 1 Visio document: $($_.Exception.Message)`n$($_.ScriptStackTrace)"
}
finally {
    if ($null -ne $document) {
        try { $document.Close() } catch {}
    }
    foreach ($shape in $modules) { Release-VisioComObject -ComObject $shape }
    foreach ($shape in $illustrationGroups) { Release-VisioComObject -ComObject $shape }
    Release-VisioComObject -ComObject $decision
    Release-VisioComObject -ComObject $terminator
    foreach ($layer in $layersByName.Values) { Release-VisioComObject -ComObject $layer }
    Release-VisioComObject -ComObject $page
    if ($null -ne $workspace) {
        $workspace.Page = $null
        $workspace.Document = $null
    }
    Release-VisioComObject -ComObject $document
    Stop-VisioSession -Session $session
}
