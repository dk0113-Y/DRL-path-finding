param(
    [switch]$ValidateOnly,
    [string]$ValidationPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDirectory "visio_common.ps1")

function Write-VisioStage {
    param([Parameter(Mandatory = $true)][string]$Stage)
    Write-Host "visio_stage=$Stage"
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
        [double]$LineWeightPt = 0.65,
        [double]$FillTransparencyPercent = 0.0
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
        -LineWeightPt $LineWeightPt `
        -FillTransparencyPercent $FillTransparencyPercent
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

function Add-Fig1MiniGrid {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)]$Layer,
        [Parameter(Mandatory = $true)][string]$NamePrefix,
        [Parameter(Mandatory = $true)][double]$Left,
        [Parameter(Mandatory = $true)][double]$Bottom,
        [Parameter(Mandatory = $true)][double]$CellSize,
        [Parameter(Mandatory = $true)][string[]]$Pattern
    )

    $colorByCode = @{
        U = "#B4BCC4"
        F = "#F8F8F6"
        O = "#1E1E1E"
        G = "#99C290"
        P = "#8281B9"
    }
    $rowCount = [int]$Pattern.Count
    $colCount = [int]$Pattern[0].Length
    $shapes = @()
    for ($row = 0; $row -lt $rowCount; $row++) {
        if ([int]$Pattern[$row].Length -ne $colCount) {
            throw "Grid pattern rows must have equal lengths."
        }
        for ($col = 0; $col -lt $colCount; $col++) {
            $code = [string]$Pattern[$row][$col]
            if (-not $colorByCode.ContainsKey($code)) {
                throw "Unknown grid color code '$code'."
            }
            $cellLeft = $Left + ($col * $CellSize)
            $cellBottom = $Bottom + (($rowCount - 1 - $row) * $CellSize)
            $shapes += New-GridCell `
                -Page $Page `
                -Layer $Layer `
                -Name ("{0}_r{1:D2}_c{2:D2}" -f $NamePrefix, $row, $col) `
                -Left $cellLeft `
                -Bottom $cellBottom `
                -Right ($cellLeft + $CellSize) `
                -Top ($cellBottom + $CellSize) `
                -FillColor ([string]$colorByCode[$code])
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
            -LineColor "#D7DDE2" `
            -LineWeightPt 0.28
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
            -LineColor "#D7DDE2" `
            -LineWeightPt 0.28
    }
    return $shapes
}

function Add-Fig1RobotParts {
    param(
        [Parameter(Mandatory = $true)]$Page,
        [Parameter(Mandatory = $true)]$Layer,
        [Parameter(Mandatory = $true)][string]$NamePrefix,
        [Parameter(Mandatory = $true)][double]$CenterX,
        [Parameter(Mandatory = $true)][double]$CenterY,
        [double]$Scale = 1.0,
        [double]$TransparencyPercent = 0.0
    )

    $parts = @()
    $bodyWidth = 0.23 * $Scale
    $bodyHeight = 0.27 * $Scale
    foreach ($offsetX in @((-0.145 * $Scale), (0.145 * $Scale))) {
        $wheel = New-Fig1Rectangle `
            -Page $Page `
            -Layer $Layer `
            -Name ("{0}_wheel_{1}" -f $NamePrefix, $(if ($offsetX -lt 0) { "left" } else { "right" })) `
            -CenterX ($CenterX + $offsetX) `
            -CenterY $CenterY `
            -Width (0.045 * $Scale) `
            -Height (0.18 * $Scale) `
            -FillColor "#30363B" `
            -LineColor "#30363B" `
            -LineWeightPt 0.35 `
            -RoundingIn (0.012 * $Scale) `
            -FillTransparencyPercent $TransparencyPercent
        Set-VisioCellFormula -Shape $wheel -CellName "LineColorTrans" -Formula "$TransparencyPercent%" -Optional
        $parts += $wheel
    }
    $body = New-Fig1Rectangle `
        -Page $Page `
        -Layer $Layer `
        -Name "${NamePrefix}_body" `
        -CenterX $CenterX `
        -CenterY $CenterY `
        -Width $bodyWidth `
        -Height $bodyHeight `
        -FillColor "#55966B" `
        -LineColor "#2F5940" `
        -LineWeightPt 0.55 `
        -RoundingIn (0.04 * $Scale) `
        -FillTransparencyPercent $TransparencyPercent
    Set-VisioCellFormula -Shape $body -CellName "LineColorTrans" -Formula "$TransparencyPercent%" -Optional
    $parts += $body

    $radar = New-Fig1Oval `
        -Page $Page `
        -Layer $Layer `
        -Name "${NamePrefix}_radar" `
        -CenterX $CenterX `
        -CenterY $CenterY `
        -Width (0.085 * $Scale) `
        -Height (0.085 * $Scale) `
        -FillColor "#E99D4E" `
        -LineColor "#2F5940" `
        -LineWeightPt 0.45 `
        -FillTransparencyPercent $TransparencyPercent
    Set-VisioCellFormula -Shape $radar -CellName "LineColorTrans" -Formula "$TransparencyPercent%" -Optional
    $parts += $radar

    $heading = New-ArrowShape `
        -Page $Page `
        -Layer $Layer `
        -Name "${NamePrefix}_heading" `
        -BeginX $CenterX `
        -BeginY ($CenterY + (0.015 * $Scale)) `
        -EndX $CenterX `
        -EndY ($CenterY + (0.105 * $Scale)) `
        -LineColor "#F8F8F6" `
        -LineWeightPt 0.60 `
        -EndArrow 13 `
        -EndArrowSize 1
    Set-VisioCellFormula -Shape $heading -CellName "LineColorTrans" -Formula "$TransparencyPercent%" -Optional
    $parts += $heading
    return $parts
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
$outputPath = $requestedOutputPath
$pngPath = $requestedPngPath

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
    "局部传感观测",
    "oₜ",
    "累计信念地图更新",
    "Bₜ",
    "策略状态构造",
    "sₜ",
    "Double DQN在线Q网络",
    "Q(sₜ,a;θ)",
    "合法动作掩码与动作选择",
    "aₜ",
    "环境执行与交互反馈",
    "rₜ, oₜ₊₁, dₜ",
    "满足终止条件？",
    "否",
    "是",
    "回合结束"
)
$forbiddenTexts = @(
    "经验回放",
    "目标网络",
    "loss",
    "参数更新",
    "n步回报"
)

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
            -ActionArrowGroupName "fig1_action_arrows_group"
        $validationDocument.Close()
        Release-VisioComObject -ComObject $validationDocument
        $validationDocument = $null
        $validationResult | ConvertTo-Json -Depth 5
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
        -PageWidthIn 14.60 `
        -PageHeightIn 5.60 `
        -PageName "Fig1_Online_Interaction"
    $document = $workspace.Document
    $page = $workspace.Page
    $layersByName["Illustrations"] = New-VisioLayer -Page $page -Name "Module_Illustrations"
    Write-VisioStage -Stage "document_created"

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
        -Page $page `
        -Name $nodeNames[6] `
        -Text "满足终止条件？" `
        -CenterX 13.65 `
        -CenterY $centerY `
        -Width 1.35 `
        -Height 1.38 `
        -FillColor $orangeFill
    $terminator = Add-VisioTerminator `
        -Page $page `
        -Name $nodeNames[7] `
        -Text "回合结束" `
        -CenterX 13.55 `
        -CenterY 1.02 `
        -Width 1.55 `
        -Height 0.58 `
        -FillColor "#EEE7DD"
    Write-VisioStage -Stage "nodes_created"

    # 1. Local observation: three-state grid, same robot palette as Figure 2, no radar rays.
    $members = @()
    $members += Add-Fig1MiniGrid `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_local_obs_grid" `
        -Left 0.50 `
        -Bottom 2.55 `
        -CellSize 0.20 `
        -Pattern @("UUUUU", "UFFOU", "UFFFO", "OUFFU", "UUUUU")
    $rangeBoundary = $page.DrawOval(0.54, 2.59, 1.46, 3.51)
    Set-VisioShapeName -Shape $rangeBoundary -Name "fig1_local_observation_range"
    Set-VisioCellFormula -Shape $rangeBoundary -CellName "FillPattern" -Formula "0"
    Set-VisioLineStyle `
        -Shape $rangeBoundary `
        -LineColor "#5185C0" `
        -LineWeightPt 0.50 `
        -TransparencyPercent 70.0
    Set-VisioCellFormula -Shape $rangeBoundary -CellName "LinePattern" -Formula "2"
    Add-ShapeToVisioLayer -Layer $layersByName["Illustrations"] -Shape $rangeBoundary
    $members += $rangeBoundary
    $members += Add-Fig1RobotParts `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_local_robot" `
        -CenterX 1.00 `
        -CenterY 3.05 `
        -Scale 0.82
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -Members $members `
        -Name $illustrationNames[0]

    # 2. Belief update: B_(t-1) + o_t -> B_t, with the robot fixed at p_t.
    $members = @()
    $members += Add-Fig1MiniGrid `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_belief_old" `
        -Left 2.12 `
        -Bottom 2.57 `
        -CellSize 0.17 `
        -Pattern @("UUU", "UFF", "UOF")
    $members += Add-Fig1MiniGrid `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_belief_observation" `
        -Left 2.77 `
        -Bottom 3.25 `
        -CellSize 0.13 `
        -Pattern @("FF", "FO")
    $members += Add-Fig1MiniGrid `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_belief_new" `
        -Left 3.17 `
        -Bottom 2.57 `
        -CellSize 0.17 `
        -Pattern @("UGG", "UFF", "UOF")
    $members += Add-Fig1RobotParts `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_belief_robot_before" `
        -CenterX 2.375 `
        -CenterY 2.825 `
        -Scale 0.42 `
        -TransparencyPercent 28.0
    $members += Add-Fig1RobotParts `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_belief_robot_after" `
        -CenterX 3.425 `
        -CenterY 2.825 `
        -Scale 0.42
    $members += Add-Fig1Text -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_belief_old_label" -Text "Bₜ₋₁" -CenterX 2.375 -CenterY 3.18 -Width 0.48 -Height 0.18 -FontSizePt 6.5
    $members += Add-Fig1Text -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_belief_observation_label" -Text "oₜ" -CenterX 2.90 -CenterY 3.62 -Width 0.28 -Height 0.16 -FontSizePt 6.5
    $members += Add-Fig1Text -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_belief_new_label" -Text "Bₜ" -CenterX 3.425 -CenterY 3.18 -Width 0.40 -Height 0.18 -FontSizePt 6.5
    $members += New-ArrowShape -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_belief_fusion_arrow" -BeginX 2.66 -BeginY 2.83 -EndX 3.10 -EndY 2.83 -LineColor "#5185C0" -LineWeightPt 1.15 -EndArrow 13 -EndArrowSize 2
    $members += New-ArrowShape -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_belief_observation_input" -BeginX 2.90 -BeginY 3.22 -EndX 2.90 -EndY 2.94 -LineColor "#5185C0" -LineWeightPt 0.85 -EndArrow 13 -EndArrowSize 1
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -Members $members `
        -Name $illustrationNames[1]

    # 3. Policy state construction: three light inputs converge to layered features.
    $members = @()
    $members += New-Fig1Rectangle -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_state_map_input" -CenterX 4.18 -CenterY 3.43 -Width 0.34 -Height 0.25 -FillColor "#DCECF7" -LineColor "#5185C0" -RoundingIn 0.02
    $members += New-Fig1Oval -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_state_position_input" -CenterX 4.18 -CenterY 3.06 -Width 0.16 -Height 0.16 -FillColor "#C96144" -LineColor "#9A4634" -LineWeightPt 0.50
    foreach ($offset in @(-0.12, 0.0, 0.12)) {
        $members += New-Fig1Oval -Page $page -Layer $layersByName["Illustrations"] -Name ("fig1_state_history_{0}" -f ([Math]::Round(($offset + 0.12) * 100))) -CenterX (4.18 + $offset) -CenterY (2.70 + (($offset + 0.12) * 0.40)) -Width 0.085 -Height 0.085 -FillColor "#8EA9D4" -LineColor "#5185C0" -LineWeightPt 0.35
    }
    foreach ($targetY in @(3.43, 3.06, 2.75)) {
        $members += New-ArrowShape -Page $page -Layer $layersByName["Illustrations"] -Name ("fig1_state_merge_{0}" -f ([Math]::Round($targetY * 100))) -BeginX 4.39 -BeginY $targetY -EndX 4.67 -EndY 3.07 -LineColor "#99AABB" -LineWeightPt 0.70 -EndArrow 13 -EndArrowSize 1
    }
    $stackColors = @("#DCECF7", "#C0BEDC", "#99C290")
    for ($index = 0; $index -lt 3; $index++) {
        $members += New-Fig1Rectangle `
            -Page $page `
            -Layer $layersByName["Illustrations"] `
            -Name ("fig1_state_feature_plane_{0}" -f ($index + 1)) `
            -CenterX (4.92 + ($index * 0.07)) `
            -CenterY (2.94 + ($index * 0.09)) `
            -Width 0.72 `
            -Height 0.48 `
            -FillColor $stackColors[$index] `
            -LineColor "#5D7183" `
            -RoundingIn 0.035 `
            -FillTransparencyPercent 12.0
    }
    $members += Add-Fig1Text -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_state_stack_label" -Text "encoded state" -CenterX 5.08 -CenterY 3.46 -Width 0.80 -Height 0.18 -FontName "Arial" -FontSizePt 5.8 -TextColor "#5D7183"
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -Members $members `
        -Name $illustrationNames[2]

    # 4. Online Q network only: state input, compact hidden layers, and eight action values.
    $members = @()
    $members += New-Fig1Rectangle -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_q_state_input" -CenterX 6.10 -CenterY 3.08 -Width 0.34 -Height 0.52 -FillColor "#DCECF7" -LineColor "#5185C0" -Text "sₜ" -FontSizePt 7.2
    foreach ($column in 0..1) {
        foreach ($row in 0..2) {
            $members += New-Fig1Oval `
                -Page $page `
                -Layer $layersByName["Illustrations"] `
                -Name ("fig1_q_hidden_c{0}_r{1}" -f ($column + 1), ($row + 1)) `
                -CenterX (6.54 + ($column * 0.37)) `
                -CenterY (2.74 + ($row * 0.34)) `
                -Width 0.15 `
                -Height 0.15 `
                -FillColor $(if ($column -eq 0) { "#8EA9D4" } else { "#99C290" }) `
                -LineColor "#5D7183" `
                -LineWeightPt 0.45
        }
    }
    foreach ($row in 0..2) {
        $members += New-LineShape -Page $page -Layer $layersByName["Illustrations"] -Name ("fig1_q_link_input_{0}" -f ($row + 1)) -BeginX 6.28 -BeginY 3.08 -EndX 6.46 -EndY (2.74 + ($row * 0.34)) -LineColor "#99AABB" -LineWeightPt 0.45 -TransparencyPercent 20.0
        $members += New-LineShape -Page $page -Layer $layersByName["Illustrations"] -Name ("fig1_q_link_hidden_{0}" -f ($row + 1)) -BeginX 6.62 -BeginY (2.74 + ($row * 0.34)) -EndX 6.83 -EndY (2.74 + ($row * 0.34)) -LineColor "#99AABB" -LineWeightPt 0.45 -TransparencyPercent 20.0
    }
    for ($index = 0; $index -lt 8; $index++) {
        $outputX = 7.30 + (($index % 2) * 0.18)
        $outputY = 2.68 + ([Math]::Floor($index / 2) * 0.25)
        $members += New-Fig1Rectangle `
            -Page $page `
            -Layer $layersByName["Illustrations"] `
            -Name ("fig1_q_output_{0:D2}" -f ($index + 1)) `
            -CenterX $outputX `
            -CenterY $outputY `
            -Width 0.13 `
            -Height 0.13 `
            -FillColor $(if ($index -eq 7) { "#C96144" } else { "#F2CB9F" }) `
            -LineColor "#8D6E58" `
            -LineWeightPt 0.35 `
            -RoundingIn 0.012
    }
    $members += New-ArrowShape -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_q_output_link" -BeginX 7.00 -BeginY 3.08 -EndX 7.20 -EndY 3.08 -LineColor "#5D7183" -LineWeightPt 0.85 -EndArrow 13 -EndArrowSize 1
    $members += Add-Fig1Text -Page $page -Layer $layersByName["Illustrations"] -Name "fig1_q_output_label" -Text "Q(a)" -CenterX 7.40 -CenterY 3.72 -Width 0.42 -Height 0.17 -FontSizePt 6.2
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -Members $members `
        -Name $illustrationNames[3]

    # 5. Action masking and selection: exactly equal Euclidean arrow lengths.
    $actionContextMembers = @()
    $actionContextMembers += Add-Fig1MiniGrid `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_action_grid" `
        -Left 8.65 `
        -Bottom 2.55 `
        -CellSize 0.20 `
        -Pattern @("UUUUU", "UFOFU", "UFFFO", "UFFOU", "UUUUU")
    $actionContextMembers += Add-Fig1RobotParts `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_action_robot" `
        -CenterX 9.15 `
        -CenterY 3.05 `
        -Scale 0.78
    $actionContextGroup = Complete-Fig1IllustrationGroup `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -Members $actionContextMembers `
        -Name "fig1_action_context_group"
    $illustrationGroups += $actionContextGroup

    $directions = @(
        [pscustomobject]@{ Name = "N"; X = 0.0; Y = 1.0; State = "legal" },
        [pscustomobject]@{ Name = "NE"; X = 1.0; Y = 1.0; State = "legal" },
        [pscustomobject]@{ Name = "E"; X = 1.0; Y = 0.0; State = "illegal" },
        [pscustomobject]@{ Name = "SE"; X = 1.0; Y = -1.0; State = "illegal" },
        [pscustomobject]@{ Name = "S"; X = 0.0; Y = -1.0; State = "legal" },
        [pscustomobject]@{ Name = "SW"; X = -1.0; Y = -1.0; State = "legal" },
        [pscustomobject]@{ Name = "W"; X = -1.0; Y = 0.0; State = "legal" },
        [pscustomobject]@{ Name = "NW"; X = -1.0; Y = 1.0; State = "selected" }
    )
    $actionArrows = @()
    $actionStartRadius = 0.12
    $actionEndRadius = $actionStartRadius + (1.50 * 0.20)
    foreach ($direction in $directions) {
        $norm = [Math]::Sqrt(
            ([double]$direction.X * [double]$direction.X) +
            ([double]$direction.Y * [double]$direction.Y)
        )
        $unitX = [double]$direction.X / $norm
        $unitY = [double]$direction.Y / $norm
        $lineColor = switch ([string]$direction.State) {
            "selected" { "#C96144" }
            "illegal" { "#99AABB" }
            default { "#5185C0" }
        }
        $actionArrows += New-ArrowShape `
            -Page $page `
            -Layer $layersByName["Illustrations"] `
            -Name ("fig1_action_arrow_{0}" -f ([string]$direction.Name)) `
            -BeginX (9.15 + ($unitX * $actionStartRadius)) `
            -BeginY (3.05 + ($unitY * $actionStartRadius)) `
            -EndX (9.15 + ($unitX * $actionEndRadius)) `
            -EndY (3.05 + ($unitY * $actionEndRadius)) `
            -LineColor $lineColor `
            -LineWeightPt $(if ([string]$direction.State -eq "selected") { 1.65 } else { 1.10 }) `
            -EndArrow 13 `
            -EndArrowSize 2
    }
    $actionArrowGroup = New-VisioShapeGroup `
        -Page $page `
        -Shapes $actionArrows `
        -Name $illustrationNames[4] `
        -Layer $layersByName["Illustrations"]
    foreach ($shape in $actionArrows) {
        Release-VisioComObject -ComObject $shape
    }
    $illustrationGroups += $actionArrowGroup
    $actionArrowGroup.BringToFront()

    # 6. Environment interaction: old/new pose only; feedback returned but not fused.
    $members = @()
    $members += Add-Fig1MiniGrid `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_environment_grid" `
        -Left 10.72 `
        -Bottom 2.55 `
        -CellSize 0.20 `
        -Pattern @("UUUUU", "UFFOU", "UFFFO", "OUFFU", "UUUUU")
    $members += Add-Fig1RobotParts `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_environment_robot_old" `
        -CenterX 11.42 `
        -CenterY 2.85 `
        -Scale 0.72 `
        -TransparencyPercent 62.0
    $members += Add-Fig1RobotParts `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -NamePrefix "fig1_environment_robot_new" `
        -CenterX 11.22 `
        -CenterY 3.05 `
        -Scale 0.72
    $feedback = @(
        [pscustomobject]@{ Text = "rₜ"; Y = 3.43; Fill = "#F2CB9F" },
        [pscustomobject]@{ Text = "oₜ₊₁"; Y = 3.10; Fill = "#DCECF7" },
        [pscustomobject]@{ Text = "dₜ"; Y = 2.77; Fill = "#C0BEDC" }
    )
    for ($index = 0; $index -lt $feedback.Count; $index++) {
        $members += New-Fig1Rectangle `
            -Page $page `
            -Layer $layersByName["Illustrations"] `
            -Name ("fig1_environment_feedback_{0:D2}" -f ($index + 1)) `
            -CenterX 12.13 `
            -CenterY ([double]$feedback[$index].Y) `
            -Width 0.48 `
            -Height 0.24 `
            -FillColor ([string]$feedback[$index].Fill) `
            -LineColor "#8A98A5" `
            -LineWeightPt 0.45 `
            -RoundingIn 0.055 `
            -Text ([string]$feedback[$index].Text) `
            -FontSizePt 6.6
    }
    $illustrationGroups += Complete-Fig1IllustrationGroup `
        -Page $page `
        -Layer $layersByName["Illustrations"] `
        -Members $members `
        -Name $illustrationNames[5]
    Write-VisioStage -Stage "native_illustrations_created"

    for ($index = 0; $index -lt 5; $index++) {
        $null = Add-VisioDynamicConnector `
            -Page $page `
            -Application $session.Application `
            -FromShape $modules[$index] `
            -ToShape $modules[$index + 1] `
            -Name "connector_main_$($index + 1)"
    }
    $null = Add-VisioDynamicConnector `
        -Page $page `
        -Application $session.Application `
        -FromShape $modules[5] `
        -ToShape $decision `
        -Name "connector_main_6"
    $null = Add-VisioDynamicConnector `
        -Page $page `
        -Application $session.Application `
        -FromShape $decision `
        -ToShape $modules[0] `
        -Name "connector_no_loop" `
        -FromXPercent 0.50 `
        -FromYPercent 1.00 `
        -ToXPercent 0.50 `
        -ToYPercent 1.00 `
        -Dashed `
        -LineWeightPt 1.15
    $null = Add-VisioDynamicConnector `
        -Page $page `
        -Application $session.Application `
        -FromShape $decision `
        -ToShape $terminator `
        -Name "connector_yes_end" `
        -FromXPercent 0.50 `
        -FromYPercent 0.00 `
        -ToXPercent 0.50 `
        -ToYPercent 1.00
    $branchLabel = Add-VisioTextBlock `
        -Page $page `
        -Name "branch_label_no" `
        -Text "否" `
        -CenterX 12.95 `
        -CenterY 5.38 `
        -Width 0.28 `
        -Height 0.20 `
        -FontName "Microsoft YaHei" `
        -FontSizePt 8.8
    Release-VisioComObject -ComObject $branchLabel
    $branchLabel = Add-VisioTextBlock `
        -Page $page `
        -Name "branch_label_yes" `
        -Text "是" `
        -CenterX 13.34 `
        -CenterY 2.18 `
        -Width 0.28 `
        -Height 0.20 `
        -FontName "Microsoft YaHei" `
        -FontSizePt 8.8
    Release-VisioComObject -ComObject $branchLabel
    Write-VisioStage -Stage "connectors_created"

    $null = $document.SaveAs($outputPath)
    Write-VisioStage -Stage "document_saved"
    $pngResult = Export-VisioPng -Page $page -Path $pngPath -Dpi 300
    Write-VisioStage -Stage "png_exported"
    $document.Close()

    foreach ($shape in $modules) {
        Release-VisioComObject -ComObject $shape
    }
    $modules = @()
    foreach ($shape in $illustrationGroups) {
        Release-VisioComObject -ComObject $shape
    }
    $illustrationGroups = @()
    Release-VisioComObject -ComObject $decision
    Release-VisioComObject -ComObject $terminator
    $decision = $null
    $terminator = $null
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
    $validation = ($validationOutput -join [Environment]::NewLine) | ConvertFrom-Json
    Write-VisioStage -Stage "document_reopened"
    Write-VisioStage -Stage "document_validated"
    Write-VisioStage -Stage "reopened_document_closed"

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
        PageCount = $validation.PageCount
        PageWidthIn = $validation.PageWidthIn
        PageHeightIn = $validation.PageHeightIn
        MainModuleCount = $validation.MainModuleCount
        DecisionNodeCount = $validation.DecisionNodeCount
        TerminatorNodeCount = $validation.TerminatorNodeCount
        LogicalNodeCount = $validation.LogicalNodeCount
        ConnectorCount = $validation.ConnectorCount
        MainConnectorCount = $validation.MainConnectorCount
        BranchConnectorCount = $validation.BranchConnectorCount
        IllustrationGroupCount = $validation.IllustrationGroupCount
        ActionArrowCount = $validation.ActionArrowCount
        ActionArrowLengthMinIn = $validation.ActionArrowLengthMinIn
        ActionArrowLengthMaxIn = $validation.ActionArrowLengthMaxIn
        NativeTopLevelShapeCount = $validation.NativeTopLevelShapeCount
        ForeignObjectCount = $validation.ForeignObjectCount
        ExternalLinkCount = $validation.ExternalLinkCount
        NoLoopTargets = $validation.NoLoopTargets
        YesBranchTargets = $validation.YesBranchTargets
        RequiredTextVerified = $validation.RequiredTextVerified
        ForbiddenTextVerified = $validation.ForbiddenTextVerified
    }
    $result | ConvertTo-Json -Depth 5
}
catch {
    throw "Failed to build or validate Figure 1 Visio document: $($_.Exception.Message)`n$($_.ScriptStackTrace)"
}
finally {
    if ($null -ne $document) {
        try {
            $document.Close()
        }
        catch {
        }
    }
    foreach ($shape in $modules) {
        Release-VisioComObject -ComObject $shape
    }
    foreach ($shape in $illustrationGroups) {
        Release-VisioComObject -ComObject $shape
    }
    Release-VisioComObject -ComObject $decision
    Release-VisioComObject -ComObject $terminator
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
