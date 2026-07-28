Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-VisioRgbFormula {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$HexColor
    )

    $value = $HexColor.Trim().TrimStart("#")
    if ($value.Length -ne 6) {
        throw "Expected a six-digit RGB hex color, got '$HexColor'."
    }
    $red = [Convert]::ToInt32($value.Substring(0, 2), 16)
    $green = [Convert]::ToInt32($value.Substring(2, 2), 16)
    $blue = [Convert]::ToInt32($value.Substring(4, 2), 16)
    return "RGB($red,$green,$blue)"
}

function Set-VisioCellFormula {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Shape,
        [Parameter(Mandatory = $true)]
        [string]$CellName,
        [Parameter(Mandatory = $true)]
        [string]$Formula,
        [switch]$Optional
    )

    if ([int]$Shape.CellExistsU($CellName, 0) -eq 0) {
        if ($Optional) {
            return
        }
        throw "Shape '$($Shape.NameU)' does not contain ShapeSheet cell '$CellName'."
    }
    $Shape.CellsU($CellName).FormulaU = $Formula
}

function Set-VisioShapeName {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Shape,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $Shape.NameU = $Name
}

function Set-VisioShapeFont {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Shape,
        [Parameter(Mandatory = $true)]
        [string]$PreferredFont,
        [string]$FallbackFont = "Arial"
    )

    $document = $Shape.ContainingPage.Document
    $fontId = $null
    foreach ($fontName in @($PreferredFont, $FallbackFont)) {
        try {
            $font = $document.Fonts.ItemU($fontName)
            $fontId = [int]$font.ID
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($font)
            break
        }
        catch {
            $fontId = $null
        }
    }
    if ($null -ne $fontId) {
        Set-VisioCellFormula -Shape $Shape -CellName "Char.Font" -Formula ([string]$fontId)
    }
}

function Set-VisioTextFormat {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Shape,
        [Parameter(Mandatory = $true)]
        [string]$FontName,
        [Parameter(Mandatory = $true)]
        [double]$FontSizePt,
        [ValidateSet("Left", "Center", "Right")]
        [string]$HorizontalAlign = "Center",
        [ValidateSet("Top", "Middle", "Bottom")]
        [string]$VerticalAlign = "Middle",
        [string]$TextColor = "#233746"
    )

    $horizontalCode = @{
        Left = 0
        Center = 1
        Right = 2
    }[$HorizontalAlign]
    $verticalCode = @{
        Top = 0
        Middle = 1
        Bottom = 2
    }[$VerticalAlign]

    Set-VisioShapeFont -Shape $Shape -PreferredFont $FontName
    Set-VisioCellFormula -Shape $Shape -CellName "Char.Size" -Formula "$FontSizePt pt"
    Set-VisioCellFormula -Shape $Shape -CellName "Char.Color" -Formula (ConvertTo-VisioRgbFormula $TextColor)
    Set-VisioCellFormula -Shape $Shape -CellName "Para.HorzAlign" -Formula ([string]$horizontalCode)
    Set-VisioCellFormula -Shape $Shape -CellName "VerticalAlign" -Formula ([string]$verticalCode)
}

function Set-VisioNodeStyle {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Shape,
        [Parameter(Mandatory = $true)]
        [string]$FillColor,
        [string]$LineColor = "#314D63",
        [double]$LineWeightPt = 1.35,
        [double]$RoundingIn = 0.0
    )

    Set-VisioCellFormula -Shape $Shape -CellName "FillPattern" -Formula "1"
    Set-VisioCellFormula -Shape $Shape -CellName "FillForegnd" -Formula (ConvertTo-VisioRgbFormula $FillColor)
    Set-VisioCellFormula -Shape $Shape -CellName "FillForegndTrans" -Formula "0%"
    Set-VisioCellFormula -Shape $Shape -CellName "LinePattern" -Formula "1"
    Set-VisioCellFormula -Shape $Shape -CellName "LineColor" -Formula (ConvertTo-VisioRgbFormula $LineColor)
    Set-VisioCellFormula -Shape $Shape -CellName "LineWeight" -Formula "$LineWeightPt pt"
    Set-VisioCellFormula -Shape $Shape -CellName "Rounding" -Formula "$RoundingIn in" -Optional
    Set-VisioCellFormula -Shape $Shape -CellName "ShdwPattern" -Formula "0" -Optional
}

function Start-VisioSession {
    [CmdletBinding()]
    param(
        [bool]$Visible = $false
    )

    $application = $null
    try {
        $application = New-Object -ComObject Visio.Application
        $application.Visible = [bool]$Visible
        $application.AlertResponse = 7
        return [pscustomobject]@{
            Application = $application
        }
    }
    catch {
        if ($null -ne $application) {
            try {
                $application.Quit()
            }
            catch {
            }
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($application)
        }
        throw "Unable to start Microsoft Visio through COM automation: $($_.Exception.Message)"
    }
}

function Stop-VisioSession {
    [CmdletBinding()]
    param(
        $Session
    )

    if ($null -eq $Session) {
        return
    }
    $application = $Session.Application
    try {
        if ($null -ne $application) {
            while ([int]$application.Documents.Count -gt 0) {
                try {
                    $application.Documents.Item(1).Close()
                }
                catch {
                    break
                }
            }
            $application.Quit()
        }
    }
    finally {
        if ($null -ne $application) {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($application)
        }
        $Session.Application = $null
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
    }
}

function New-VisioDocumentPage {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Session,
        [double]$PageWidthIn = 13.2,
        [double]$PageHeightIn = 4.8,
        [string]$PageName = "Fig1_Online_Interaction"
    )

    $document = $Session.Application.Documents.Add("")
    $page = $document.Pages.Item(1)
    $page.NameU = $PageName
    $page.PageSheet.CellsU("PageWidth").FormulaU = "$PageWidthIn in"
    $page.PageSheet.CellsU("PageHeight").FormulaU = "$PageHeightIn in"
    Set-VisioCellFormula -Shape $page.PageSheet -CellName "PageScale" -Formula "1 in" -Optional
    Set-VisioCellFormula -Shape $page.PageSheet -CellName "DrawingScale" -Formula "1 in" -Optional
    return [pscustomobject]@{
        Document = $document
        Page = $page
    }
}

function Add-VisioTextBlock {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Text,
        [Parameter(Mandatory = $true)]
        [double]$CenterX,
        [Parameter(Mandatory = $true)]
        [double]$CenterY,
        [Parameter(Mandatory = $true)]
        [double]$Width,
        [Parameter(Mandatory = $true)]
        [double]$Height,
        [string]$FontName = "Microsoft YaHei",
        [double]$FontSizePt = 10.0,
        [ValidateSet("Left", "Center", "Right")]
        [string]$HorizontalAlign = "Center",
        [ValidateSet("Top", "Middle", "Bottom")]
        [string]$VerticalAlign = "Middle"
    )

    $shape = $Page.DrawRectangle(
        $CenterX - ($Width / 2.0),
        $CenterY - ($Height / 2.0),
        $CenterX + ($Width / 2.0),
        $CenterY + ($Height / 2.0)
    )
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioCellFormula -Shape $shape -CellName "FillPattern" -Formula "0"
    Set-VisioCellFormula -Shape $shape -CellName "LinePattern" -Formula "0"
    $shape.Text = $Text
    Set-VisioTextFormat `
        -Shape $shape `
        -FontName $FontName `
        -FontSizePt $FontSizePt `
        -HorizontalAlign $HorizontalAlign `
        -VerticalAlign $VerticalAlign
    return ,$shape
}

function Add-VisioRoundedModule {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [string]$Formula = "",
        [Parameter(Mandatory = $true)]
        [double]$CenterX,
        [Parameter(Mandatory = $true)]
        [double]$CenterY,
        [Parameter(Mandatory = $true)]
        [double]$Width,
        [Parameter(Mandatory = $true)]
        [double]$Height,
        [Parameter(Mandatory = $true)]
        [string]$FillColor
    )

    $shape = $Page.DrawRectangle(
        $CenterX - ($Width / 2.0),
        $CenterY - ($Height / 2.0),
        $CenterX + ($Width / 2.0),
        $CenterY + ($Height / 2.0)
    )
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioNodeStyle -Shape $shape -FillColor $FillColor -RoundingIn 0.14

    $top = $CenterY + ($Height / 2.0)
    $null = Add-VisioTextBlock `
        -Page $Page `
        -Name "${Name}_title" `
        -Text $Title `
        -CenterX $CenterX `
        -CenterY ($top - 0.34) `
        -Width ($Width - 0.10) `
        -Height 0.42 `
        -FontName "Microsoft YaHei" `
        -FontSizePt 9.6

    if (-not [string]::IsNullOrWhiteSpace($Formula)) {
        $null = Add-VisioTextBlock `
            -Page $Page `
            -Name "${Name}_formula" `
            -Text $Formula `
            -CenterX $CenterX `
            -CenterY ($top - 0.83) `
            -Width ($Width - 0.12) `
            -Height 0.38 `
            -FontName "Cambria Math" `
            -FontSizePt 11.0
    }
    return ,$shape
}

function Add-VisioDiamond {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Text,
        [Parameter(Mandatory = $true)]
        [double]$CenterX,
        [Parameter(Mandatory = $true)]
        [double]$CenterY,
        [Parameter(Mandatory = $true)]
        [double]$Width,
        [Parameter(Mandatory = $true)]
        [double]$Height,
        [string]$FillColor = "#F5E7D7"
    )

    [double[]]$points = @(
        [double]$CenterX, [double]($CenterY + ($Height / 2.0)),
        [double]($CenterX + ($Width / 2.0)), [double]$CenterY,
        [double]$CenterX, [double]($CenterY - ($Height / 2.0)),
        [double]($CenterX - ($Width / 2.0)), [double]$CenterY,
        [double]$CenterX, [double]($CenterY + ($Height / 2.0))
    )
    $shape = $Page.DrawPolyline($points, 0)
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioNodeStyle -Shape $shape -FillColor $FillColor
    $shape.Text = $Text
    Set-VisioTextFormat -Shape $shape -FontName "Microsoft YaHei" -FontSizePt 9.5
    return ,$shape
}

function Add-VisioTerminator {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Text,
        [Parameter(Mandatory = $true)]
        [double]$CenterX,
        [Parameter(Mandatory = $true)]
        [double]$CenterY,
        [Parameter(Mandatory = $true)]
        [double]$Width,
        [Parameter(Mandatory = $true)]
        [double]$Height,
        [string]$FillColor = "#EEE7DD"
    )

    $shape = $Page.DrawRectangle(
        $CenterX - ($Width / 2.0),
        $CenterY - ($Height / 2.0),
        $CenterX + ($Width / 2.0),
        $CenterY + ($Height / 2.0)
    )
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioNodeStyle -Shape $shape -FillColor $FillColor -RoundingIn ($Height / 2.0)
    $shape.Text = $Text
    Set-VisioTextFormat -Shape $shape -FontName "Microsoft YaHei" -FontSizePt 10.0
    return ,$shape
}

function Add-VisioDynamicConnector {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        $Application,
        [Parameter(Mandatory = $true)]
        $FromShape,
        [Parameter(Mandatory = $true)]
        $ToShape,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [double]$FromXPercent = 1.0,
        [double]$FromYPercent = 0.5,
        [double]$ToXPercent = 0.0,
        [double]$ToYPercent = 0.5,
        [string]$Label = "",
        [switch]$Dashed,
        [string]$LineColor = "#314D63",
        [double]$LineWeightPt = 1.35
    )

    $connector = $Page.Drop($Application.ConnectorToolDataObject, 0.0, 0.0)
    Set-VisioShapeName -Shape $connector -Name $Name
    $null = $connector.CellsU("BeginX").GlueToPos(
        $FromShape,
        [double]$FromXPercent,
        [double]$FromYPercent
    )
    $null = $connector.CellsU("EndX").GlueToPos(
        $ToShape,
        [double]$ToXPercent,
        [double]$ToYPercent
    )

    Set-VisioCellFormula -Shape $connector -CellName "LineColor" -Formula (ConvertTo-VisioRgbFormula $LineColor)
    Set-VisioCellFormula -Shape $connector -CellName "LineWeight" -Formula "$LineWeightPt pt"
    Set-VisioCellFormula -Shape $connector -CellName "LinePattern" -Formula $(if ($Dashed) { "2" } else { "1" })
    Set-VisioCellFormula -Shape $connector -CellName "BeginArrow" -Formula "0"
    Set-VisioCellFormula -Shape $connector -CellName "EndArrow" -Formula "13"
    Set-VisioCellFormula -Shape $connector -CellName "EndArrowSize" -Formula "2"
    Set-VisioCellFormula -Shape $connector -CellName "ShapeRouteStyle" -Formula "1" -Optional
    Set-VisioCellFormula -Shape $connector -CellName "ConLineRouteExt" -Formula "1" -Optional

    if (-not [string]::IsNullOrWhiteSpace($Label)) {
        $connector.Text = $Label
        Set-VisioTextFormat -Shape $connector -FontName "Microsoft YaHei" -FontSizePt 9.0
    }
    return ,$connector
}

function Get-VisioShapeByName {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    try {
        return ,$Page.Shapes.ItemU($Name)
    }
    catch {
        throw "Required Visio shape '$Name' was not found."
    }
}

function Get-VisioConnectorTargetNames {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Connector
    )

    $names = New-Object System.Collections.Generic.List[string]
    $connects = $Connector.Connects
    try {
        for ($index = 1; $index -le [int]$connects.Count; $index++) {
            $connection = $connects.Item($index)
            try {
                $names.Add([string]$connection.ToSheet.NameU)
            }
            finally {
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($connection)
            }
        }
    }
    finally {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($connects)
    }
    return @($names | Sort-Object -Unique)
}

function Test-VisioOnlineInteractionDocument {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Document,
        [Parameter(Mandatory = $true)]
        [string[]]$RequiredNodeNames,
        [Parameter(Mandatory = $true)]
        [string[]]$RequiredTexts,
        [Parameter(Mandatory = $true)]
        [string]$FirstModuleName,
        [Parameter(Mandatory = $true)]
        [string]$DecisionName,
        [Parameter(Mandatory = $true)]
        [string]$TerminatorName
    )

    if ([int]$Document.Pages.Count -ne 1) {
        throw "Expected exactly one Visio page, found $($Document.Pages.Count)."
    }
    $page = $Document.Pages.Item(1)
    try {
        foreach ($nodeName in $RequiredNodeNames) {
            $shape = Get-VisioShapeByName -Page $page -Name $nodeName
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shape)
        }

        $allText = New-Object System.Collections.Generic.List[string]
        $foreignObjectCount = 0
        $externalLinkCount = 0
        $connectorNames = New-Object System.Collections.Generic.List[string]
        for ($index = 1; $index -le [int]$page.Shapes.Count; $index++) {
            $shape = $page.Shapes.Item($index)
            try {
                if (-not [string]::IsNullOrWhiteSpace([string]$shape.Text)) {
                    $allText.Add([string]$shape.Text)
                }
                if ([int]$shape.Type -eq 4) {
                    $foreignObjectCount++
                }
                try {
                    $externalLinkCount += [int]$shape.Hyperlinks.Count
                }
                catch {
                }
                if ([string]$shape.NameU -like "connector_*") {
                    $connectorNames.Add([string]$shape.NameU)
                }
            }
            finally {
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shape)
            }
        }

        $joinedText = [string]::Join("`n", $allText)
        foreach ($requiredText in $RequiredTexts) {
            if (-not $joinedText.Contains($requiredText)) {
                throw "Required Visio text '$requiredText' was not found."
            }
        }

        $expectedMainConnectors = 1..7 | ForEach-Object { "connector_main_$_" }
        foreach ($connectorName in $expectedMainConnectors) {
            if (-not $connectorNames.Contains($connectorName)) {
                throw "Required main-flow connector '$connectorName' was not found."
            }
        }
        foreach ($branchConnectorName in @("connector_no_loop", "connector_yes_end")) {
            if (-not $connectorNames.Contains($branchConnectorName)) {
                throw "Required branch connector '$branchConnectorName' was not found."
            }
        }

        $noConnector = Get-VisioShapeByName -Page $page -Name "connector_no_loop"
        $yesConnector = Get-VisioShapeByName -Page $page -Name "connector_yes_end"
        try {
            $noTargets = Get-VisioConnectorTargetNames -Connector $noConnector
            $yesTargets = Get-VisioConnectorTargetNames -Connector $yesConnector
        }
        finally {
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($noConnector)
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($yesConnector)
        }
        foreach ($name in @($DecisionName, $FirstModuleName)) {
            if ($name -notin $noTargets) {
                throw "The '否' loop connector is not glued to '$name'."
            }
        }
        foreach ($name in @($DecisionName, $TerminatorName)) {
            if ($name -notin $yesTargets) {
                throw "The '是' connector is not glued to '$name'."
            }
        }

        $dataRecordsetCount = 0
        try {
            $dataRecordsetCount = [int]$Document.DataRecordsets.Count
        }
        catch {
        }
        if ($foreignObjectCount -ne 0) {
            throw "Found $foreignObjectCount foreign image/OLE object(s); expected none."
        }
        if (($externalLinkCount + $dataRecordsetCount) -ne 0) {
            throw "Found external hyperlinks or data links; expected none."
        }

        $pageWidth = [double]$page.PageSheet.CellsU("PageWidth").ResultIU
        $pageHeight = [double]$page.PageSheet.CellsU("PageHeight").ResultIU
        return [pscustomobject]@{
            PageCount = [int]$Document.Pages.Count
            PageWidthIn = [Math]::Round($pageWidth, 3)
            PageHeightIn = [Math]::Round($pageHeight, 3)
            LogicalNodeCount = [int]$RequiredNodeNames.Count
            MainModuleCount = 7
            DecisionNodeCount = 1
            TerminatorNodeCount = 1
            ConnectorCount = [int]$connectorNames.Count
            MainConnectorCount = 7
            BranchConnectorCount = 2
            NativeTopLevelShapeCount = [int]$page.Shapes.Count
            ForeignObjectCount = [int]$foreignObjectCount
            ExternalLinkCount = [int]($externalLinkCount + $dataRecordsetCount)
            NoLoopTargets = @($noTargets)
            YesBranchTargets = @($yesTargets)
            RequiredTextVerified = $true
        }
    }
    finally {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($page)
    }
}
