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

$outputDirectory = "C:\Users\Dk\Desktop\SCI\paper_picture\visio_outputs"
$requestedOutputPath = Join-Path $outputDirectory "fig1_online_decision_interaction.vsdx"
$requestedPngPath = Join-Path $outputDirectory "fig1_online_decision_interaction.png"
$outputPath = $requestedOutputPath
$pngPath = $requestedPngPath

$session = $null
$workspace = $null
$document = $null
$page = $null
$modules = @()
$decision = $null
$terminator = $null
$pngResult = $null
$outputPair = $null

$nodeNames = @(
    "module_1_local_observation",
    "module_2_belief_update",
    "module_3_policy_state",
    "module_4_online_q_network",
    "module_5_mask_action",
    "module_6_environment_execute",
    "module_7_environment_feedback",
    "decision_terminal_condition",
    "terminator_episode_end"
)
$requiredTexts = @(
    "局部传感观测",
    "o_t",
    "累计信念地图更新",
    "B_t",
    "策略状态构造",
    "s_t",
    "Double DQN在线Q网络",
    "Q(s_t,a;θ)",
    "合法动作掩码与动作选择",
    "a_t",
    "环境执行",
    "环境反馈",
    "r_t, o_{t+1}, d_t",
    "满足终止条件？",
    "否",
    "是",
    "回合结束"
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
            -DecisionName $nodeNames[7] `
            -TerminatorName $nodeNames[8]
        $validationDocument.Close()
        Release-VisioComObject -ComObject $validationDocument
        $validationDocument = $null
        $validationResult | ConvertTo-Json -Depth 4
    }
    finally {
        if ($null -ne $validationDocument) {
            try {
                $validationDocument.Close()
            }
            catch {
            }
            Release-VisioComObject -ComObject $validationDocument
            $validationDocument = $null
        }
        Stop-VisioSession -Session $validationSession
    }
    return
}

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
        -PageWidthIn 13.2 `
        -PageHeightIn 4.8 `
        -PageName "Fig1_Online_Interaction"
    $document = $workspace.Document
    $page = $workspace.Page
    Write-VisioStage -Stage "document_created"

    $centerY = 3.30
    $moduleHeight = 2.25
    $blueFill = "#DCECF7"
    $greenFill = "#E4F1E4"
    $orangeFill = "#F5E7D7"

    $modules = @(
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[0] -Title "局部传感观测" -Formula "o_t" -CenterX 0.90 -CenterY $centerY -Width 1.25 -Height $moduleHeight -FillColor $blueFill),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[1] -Title "累计信念地图更新" -Formula "B_t" -CenterX 2.45 -CenterY $centerY -Width 1.45 -Height $moduleHeight -FillColor $blueFill),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[2] -Title "策略状态构造" -Formula "s_t" -CenterX 4.05 -CenterY $centerY -Width 1.25 -Height $moduleHeight -FillColor $blueFill),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[3] -Title "Double DQN在线Q网络" -Formula "Q(s_t,a;θ)" -CenterX 5.80 -CenterY $centerY -Width 1.60 -Height $moduleHeight -FillColor $greenFill),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[4] -Title "合法动作掩码与动作选择" -Formula "a_t" -CenterX 7.75 -CenterY $centerY -Width 1.65 -Height $moduleHeight -FillColor $greenFill),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[5] -Title "环境执行" -CenterX 9.45 -CenterY $centerY -Width 1.15 -Height $moduleHeight -FillColor $orangeFill),
        (Add-VisioRoundedModule -Page $page -Name $nodeNames[6] -Title "环境反馈" -Formula "r_t, o_{t+1}, d_t" -CenterX 11.00 -CenterY $centerY -Width 1.40 -Height $moduleHeight -FillColor $orangeFill)
    )
    $decision = Add-VisioDiamond `
        -Page $page `
        -Name $nodeNames[7] `
        -Text "满足终止条件？" `
        -CenterX 12.50 `
        -CenterY $centerY `
        -Width 1.20 `
        -Height 1.30 `
        -FillColor $orangeFill
    $terminator = Add-VisioTerminator `
        -Page $page `
        -Name $nodeNames[8] `
        -Text "回合结束" `
        -CenterX 12.35 `
        -CenterY 1.15 `
        -Width 1.45 `
        -Height 0.58 `
        -FillColor "#EEE7DD"
    Write-VisioStage -Stage "nodes_created"

    for ($index = 0; $index -lt 6; $index++) {
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
        -FromShape $modules[6] `
        -ToShape $decision `
        -Name "connector_main_7"
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
        -Label "否" `
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
        -ToYPercent 1.00 `
        -Label "是"
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
    Release-VisioComObject -ComObject $decision
    Release-VisioComObject -ComObject $terminator
    $decision = $null
    $terminator = $null
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
        NativeTopLevelShapeCount = $validation.NativeTopLevelShapeCount
        ForeignObjectCount = $validation.ForeignObjectCount
        ExternalLinkCount = $validation.ExternalLinkCount
        NoLoopTargets = $validation.NoLoopTargets
        YesBranchTargets = $validation.YesBranchTargets
        RequiredTextVerified = $validation.RequiredTextVerified
    }
    $result | ConvertTo-Json -Depth 4
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
    $modules = @()
    Release-VisioComObject -ComObject $decision
    Release-VisioComObject -ComObject $terminator
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
