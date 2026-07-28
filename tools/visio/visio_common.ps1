Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Release-VisioComObject {
    [CmdletBinding()]
    param(
        $ComObject
    )

    if ($null -eq $ComObject) {
        return
    }
    try {
        if ([Runtime.InteropServices.Marshal]::IsComObject($ComObject)) {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($ComObject)
        }
    }
    catch {
    }
}

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
        [double]$RoundingIn = 0.0,
        [double]$FillTransparencyPercent = 0.0
    )

    Set-VisioCellFormula -Shape $Shape -CellName "FillPattern" -Formula "1"
    Set-VisioCellFormula -Shape $Shape -CellName "FillForegnd" -Formula (ConvertTo-VisioRgbFormula $FillColor)
    Set-VisioCellFormula -Shape $Shape -CellName "FillForegndTrans" -Formula "$FillTransparencyPercent%"
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
        # InvisibleApp creates an isolated background Visio process instead of
        # attaching automation to a document the user may already have open.
        $progId = if ($Visible) { "Visio.Application" } else { "Visio.InvisibleApp" }
        $application = New-Object -ComObject $progId
        if ($Visible) {
            $application.Visible = $true
        }
        $application.AlertResponse = 7
        return [pscustomobject]@{
            Application = $application
            ProgId = $progId
        }
    }
    catch {
        if ($null -ne $application) {
            try {
                $application.Quit()
            }
            catch {
            }
            Release-VisioComObject -ComObject $application
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
    $documents = $null
    try {
        if ($null -ne $application) {
            $documents = $application.Documents
            while ([int]$documents.Count -gt 0) {
                $document = $null
                try {
                    $document = $documents.Item(1)
                    $document.Close()
                }
                catch {
                    break
                }
                finally {
                    Release-VisioComObject -ComObject $document
                }
            }
            $application.Quit()
        }
    }
    finally {
        Release-VisioComObject -ComObject $documents
        Release-VisioComObject -ComObject $application
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

function Test-VisioFileLocked {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $stream = $null
    try {
        $stream = [IO.File]::Open(
            $Path,
            [IO.FileMode]::Open,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
        return $false
    }
    catch {
        return $true
    }
    finally {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
    }
}

function Resolve-VisioOutputPair {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$VsdxPath,
        [Parameter(Mandatory = $true)]
        [string]$PngPath
    )

    $lockedPaths = @(
        @($VsdxPath, $PngPath) |
            Where-Object { Test-VisioFileLocked -Path $_ }
    )
    if ($lockedPaths.Count -gt 0) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $vsdxDirectory = Split-Path -Parent $VsdxPath
        $pngDirectory = Split-Path -Parent $PngPath
        $vsdxName = [IO.Path]::GetFileNameWithoutExtension($VsdxPath)
        $pngName = [IO.Path]::GetFileNameWithoutExtension($PngPath)
        return [pscustomobject]@{
            VsdxPath = Join-Path $vsdxDirectory "${vsdxName}_$timestamp.vsdx"
            PngPath = Join-Path $pngDirectory "${pngName}_$timestamp.png"
            UsedTimestampFallback = $true
            LockedPaths = @($lockedPaths)
        }
    }

    foreach ($path in @($VsdxPath, $PngPath)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
    return [pscustomobject]@{
        VsdxPath = $VsdxPath
        PngPath = $PngPath
        UsedTimestampFallback = $false
        LockedPaths = @()
    }
}

function New-VisioLayer {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $layer = $null
    try {
        $layer = $Page.Layers.ItemU($Name)
    }
    catch {
        $layer = $Page.Layers.Add($Name)
    }

    # Visio layer cell indices: Visible=4, Print=5, Active=6, Lock=7,
    # Snap=8, Glue=9.
    $layer.CellsC(4).FormulaU = "1"
    $layer.CellsC(5).FormulaU = "1"
    $layer.CellsC(6).FormulaU = "0"
    $layer.CellsC(7).FormulaU = "0"
    $layer.CellsC(8).FormulaU = "1"
    $layer.CellsC(9).FormulaU = "1"
    return ,$layer
}

function Add-ShapeToVisioLayer {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Layer,
        [Parameter(Mandatory = $true)]
        $Shape
    )

    $Layer.Add($Shape, 0)
}

function Set-VisioLineStyle {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Shape,
        [Parameter(Mandatory = $true)]
        [string]$LineColor,
        [double]$LineWeightPt = 1.0,
        [double]$TransparencyPercent = 0.0,
        [int]$BeginArrow = 0,
        [int]$EndArrow = 0,
        [int]$EndArrowSize = 2
    )

    Set-VisioCellFormula -Shape $Shape -CellName "LinePattern" -Formula "1"
    Set-VisioCellFormula -Shape $Shape -CellName "LineColor" -Formula (ConvertTo-VisioRgbFormula $LineColor)
    Set-VisioCellFormula -Shape $Shape -CellName "LineWeight" -Formula "$LineWeightPt pt"
    Set-VisioCellFormula -Shape $Shape -CellName "LineColorTrans" -Formula "$TransparencyPercent%" -Optional
    Set-VisioCellFormula -Shape $Shape -CellName "BeginArrow" -Formula ([string]$BeginArrow)
    Set-VisioCellFormula -Shape $Shape -CellName "EndArrow" -Formula ([string]$EndArrow)
    Set-VisioCellFormula -Shape $Shape -CellName "EndArrowSize" -Formula ([string]$EndArrowSize)
}

function New-GridCell {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        $Layer,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [double]$Left,
        [Parameter(Mandatory = $true)]
        [double]$Bottom,
        [Parameter(Mandatory = $true)]
        [double]$Right,
        [Parameter(Mandatory = $true)]
        [double]$Top,
        [Parameter(Mandatory = $true)]
        [string]$FillColor
    )

    $shape = $Page.DrawRectangle($Left, $Bottom, $Right, $Top)
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioCellFormula -Shape $shape -CellName "FillPattern" -Formula "1"
    Set-VisioCellFormula -Shape $shape -CellName "FillForegnd" -Formula (ConvertTo-VisioRgbFormula $FillColor)
    Set-VisioCellFormula -Shape $shape -CellName "FillForegndTrans" -Formula "0%"
    Set-VisioCellFormula -Shape $shape -CellName "LinePattern" -Formula "0"
    Set-VisioCellFormula -Shape $shape -CellName "ShdwPattern" -Formula "0" -Optional
    Add-ShapeToVisioLayer -Layer $Layer -Shape $shape
    return ,$shape
}

function New-GridLine {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        $Layer,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [double]$BeginX,
        [Parameter(Mandatory = $true)]
        [double]$BeginY,
        [Parameter(Mandatory = $true)]
        [double]$EndX,
        [Parameter(Mandatory = $true)]
        [double]$EndY,
        [string]$LineColor = "#C8D2DA",
        [double]$LineWeightPt = 0.55
    )

    $shape = $Page.DrawLine($BeginX, $BeginY, $EndX, $EndY)
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioLineStyle -Shape $shape -LineColor $LineColor -LineWeightPt $LineWeightPt
    Add-ShapeToVisioLayer -Layer $Layer -Shape $shape
    return ,$shape
}

function New-LineShape {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        $Layer,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [double]$BeginX,
        [Parameter(Mandatory = $true)]
        [double]$BeginY,
        [Parameter(Mandatory = $true)]
        [double]$EndX,
        [Parameter(Mandatory = $true)]
        [double]$EndY,
        [Parameter(Mandatory = $true)]
        [string]$LineColor,
        [double]$LineWeightPt = 0.75,
        [double]$TransparencyPercent = 0.0
    )

    $shape = $Page.DrawLine($BeginX, $BeginY, $EndX, $EndY)
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioLineStyle `
        -Shape $shape `
        -LineColor $LineColor `
        -LineWeightPt $LineWeightPt `
        -TransparencyPercent $TransparencyPercent
    Add-ShapeToVisioLayer -Layer $Layer -Shape $shape
    return ,$shape
}

function New-ArrowShape {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        $Layer,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [double]$BeginX,
        [Parameter(Mandatory = $true)]
        [double]$BeginY,
        [Parameter(Mandatory = $true)]
        [double]$EndX,
        [Parameter(Mandatory = $true)]
        [double]$EndY,
        [Parameter(Mandatory = $true)]
        [string]$LineColor,
        [double]$LineWeightPt = 2.2,
        [int]$EndArrow = 13,
        [int]$EndArrowSize = 3
    )

    $shape = $Page.DrawLine($BeginX, $BeginY, $EndX, $EndY)
    Set-VisioShapeName -Shape $shape -Name $Name
    Set-VisioLineStyle `
        -Shape $shape `
        -LineColor $LineColor `
        -LineWeightPt $LineWeightPt `
        -EndArrow $EndArrow `
        -EndArrowSize $EndArrowSize
    Add-ShapeToVisioLayer -Layer $Layer -Shape $shape
    return ,$shape
}

function New-VisioShapeGroup {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [object[]]$Shapes,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        $Layer
    )

    if ($Shapes.Count -lt 1) {
        throw "Cannot create Visio group '$Name' without member shapes."
    }

    $selection = $null
    try {
        # visSelTypeEmpty=0, visSelModeSkipSuper=256, visSelect=2.
        $selection = $Page.CreateSelection(0, 256, $null)
        foreach ($shape in $Shapes) {
            $selection.Select($shape, 2)
        }
        $group = $selection.Group()
        Set-VisioShapeName -Shape $group -Name $Name
        if ($null -ne $Layer) {
            Add-ShapeToVisioLayer -Layer $Layer -Shape $group
        }
        return ,$group
    }
    finally {
        Release-VisioComObject -ComObject $selection
    }
}

function Set-PageDrawingSize {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [double]$WidthIn,
        [Parameter(Mandatory = $true)]
        [double]$HeightIn
    )

    $Page.PageSheet.CellsU("PageWidth").FormulaU = "$WidthIn in"
    $Page.PageSheet.CellsU("PageHeight").FormulaU = "$HeightIn in"
}

function Export-VisioPng {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Page,
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [int]$Dpi = 300
    )

    $settings = $null
    try {
        $pageWidth = [double]$Page.PageSheet.CellsU("PageWidth").ResultIU
        $pageHeight = [double]$Page.PageSheet.CellsU("PageHeight").ResultIU
        $pixelWidth = [Math]::Max(1, [int][Math]::Round($pageWidth * $Dpi))
        $pixelHeight = [Math]::Max(1, [int][Math]::Round($pageHeight * $Dpi))

        $settings = $Page.Application.Settings
        # visRasterUseCustomResolution=3, visRasterPixelsPerInch=0.
        $settings.SetRasterExportResolution(3, [double]$Dpi, [double]$Dpi, 0)
        # visRasterFitToCustomSize=3, visRasterPixel=0.
        $settings.SetRasterExportSize(3, [double]$pixelWidth, [double]$pixelHeight, 0)
        # 24-bit color, opaque white background.
        $settings.RasterExportColorFormat = 3
        $settings.RasterExportBackgroundColor = 16777215
        $settings.RasterExportUseTransparencyColor = $false
        $settings.RasterExportRotation = 0
        $settings.RasterExportFlip = 0
        $settings.RasterExportQuality = 100
        $Page.Export($Path)
    }
    finally {
        Release-VisioComObject -ComObject $settings
    }

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Visio PNG export was not created at '$Path'."
    }
    $file = Get-Item -LiteralPath $Path
    if ([long]$file.Length -le 0) {
        throw "Visio PNG export is empty: '$Path'."
    }
    return [pscustomobject]@{
        Path = $file.FullName
        FileSizeBytes = [long]$file.Length
        PixelWidth = $pixelWidth
        PixelHeight = $pixelHeight
        Dpi = $Dpi
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
        [string]$VerticalAlign = "Middle",
        [string]$TextColor = "#233746"
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
        -VerticalAlign $VerticalAlign `
        -TextColor $TextColor
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
        [string]$FillColor,
        [double]$TitleFontSizePt = 9.6,
        [double]$FormulaFontSizePt = 11.0
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
        -FontSizePt $TitleFontSizePt

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
            -FontSizePt $FormulaFontSizePt
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
        [string]$TerminatorName,
        [int]$ExpectedMainModuleCount = 6,
        [int]$ExpectedMainConnectorCount = 6,
        [string[]]$RequiredIllustrationNames = @(),
        [string[]]$ForbiddenTexts = @(),
        [string]$ActionArrowGroupName = ""
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
        foreach ($illustrationName in $RequiredIllustrationNames) {
            $shape = Get-VisioShapeByName -Page $page -Name $illustrationName
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
        foreach ($forbiddenText in $ForbiddenTexts) {
            if ($joinedText.Contains($forbiddenText)) {
                throw "Forbidden training-related Visio text '$forbiddenText' was found."
            }
        }

        $expectedMainConnectors = 1..$ExpectedMainConnectorCount | ForEach-Object { "connector_main_$_" }
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

        $actionArrowCount = 0
        $actionArrowLengthMin = 0.0
        $actionArrowLengthMax = 0.0
        if (-not [string]::IsNullOrWhiteSpace($ActionArrowGroupName)) {
            $actionGroup = Get-VisioShapeByName -Page $page -Name $ActionArrowGroupName
            try {
                $actionLengths = New-Object System.Collections.Generic.List[double]
                for ($index = 1; $index -le [int]$actionGroup.Shapes.Count; $index++) {
                    $arrow = $actionGroup.Shapes.Item($index)
                    try {
                        if ([string]$arrow.NameU -notlike "fig1_action_arrow_*") {
                            continue
                        }
                        $beginX = [double]$arrow.CellsU("BeginX").ResultIU
                        $beginY = [double]$arrow.CellsU("BeginY").ResultIU
                        $endX = [double]$arrow.CellsU("EndX").ResultIU
                        $endY = [double]$arrow.CellsU("EndY").ResultIU
                        $actionLengths.Add(
                            [Math]::Sqrt(
                                (($endX - $beginX) * ($endX - $beginX)) +
                                (($endY - $beginY) * ($endY - $beginY))
                            )
                        )
                    }
                    finally {
                        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($arrow)
                    }
                }
                $actionArrowCount = [int]$actionLengths.Count
                if ($actionArrowCount -ne 8) {
                    throw "Expected 8 normalized action arrows, found $actionArrowCount."
                }
                $actionArrowLengthMin = [double]($actionLengths | Measure-Object -Minimum).Minimum
                $actionArrowLengthMax = [double]($actionLengths | Measure-Object -Maximum).Maximum
                if (($actionArrowLengthMax - $actionArrowLengthMin) -gt 0.01) {
                    throw (
                        "Action-arrow lengths are inconsistent: min=$actionArrowLengthMin, " +
                        "max=$actionArrowLengthMax."
                    )
                }
            }
            finally {
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($actionGroup)
            }
        }

        $pageWidth = [double]$page.PageSheet.CellsU("PageWidth").ResultIU
        $pageHeight = [double]$page.PageSheet.CellsU("PageHeight").ResultIU
        return [pscustomobject]@{
            PageCount = [int]$Document.Pages.Count
            PageWidthIn = [Math]::Round($pageWidth, 3)
            PageHeightIn = [Math]::Round($pageHeight, 3)
            LogicalNodeCount = [int]$RequiredNodeNames.Count
            MainModuleCount = [int]$ExpectedMainModuleCount
            DecisionNodeCount = 1
            TerminatorNodeCount = 1
            ConnectorCount = [int]$connectorNames.Count
            MainConnectorCount = [int]$ExpectedMainConnectorCount
            BranchConnectorCount = 2
            IllustrationGroupCount = [int]$RequiredIllustrationNames.Count
            ActionArrowCount = [int]$actionArrowCount
            ActionArrowLengthMinIn = [Math]::Round($actionArrowLengthMin, 4)
            ActionArrowLengthMaxIn = [Math]::Round($actionArrowLengthMax, 4)
            NativeTopLevelShapeCount = [int]$page.Shapes.Count
            ForeignObjectCount = [int]$foreignObjectCount
            ExternalLinkCount = [int]($externalLinkCount + $dataRecordsetCount)
            NoLoopTargets = @($noTargets)
            YesBranchTargets = @($yesTargets)
            RequiredTextVerified = $true
            ForbiddenTextVerified = $true
        }
    }
    finally {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($page)
    }
}
