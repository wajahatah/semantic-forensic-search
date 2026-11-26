# Finishes the Word report built by scripts/report_template_build.js (D-080): replaces the @@TOC@@ / @@LOF@@ / @@LOT@@
# markers with a table of contents and the lists of figures and tables, updates every field (captions, page numbers,
# the STYLEREF header), saves the .docx and exports the PDF. Uses its own hidden Word instance; a Word window the
# user has open is not touched.
#
#   powershell -ExecutionPolicy Bypass -File scripts\report_word_finish.ps1 -Docx report\Ahmed_Wajahat_DLReport.docx -Pdf report\Ahmed_Wajahat_DLReport.pdf
param(
    [Parameter(Mandatory = $true)][string]$Docx,
    [Parameter(Mandatory = $true)][string]$Pdf
)
$ErrorActionPreference = "Stop"
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::GetFullPath($Pdf)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $false, $false)

    function Set-List([string]$mark, [string]$kind) {
        $r = $doc.Content
        $f = $r.Find
        $f.ClearFormatting()
        if (-not $f.Execute($mark)) { throw "marker $mark not found" }
        $r.Text = ""
        if ($kind -eq "toc") {
            # Range, UseHeadingStyles, UpperHeadingLevel, LowerHeadingLevel
            [void]$doc.TablesOfContents.Add($r, $true, 1, 2)
        } else {
            # Range, Caption label, IncludeLabel
            [void]$doc.TablesOfFigures.Add($r, $kind, $true)
        }
    }
    Set-List "@@TOC@@" "toc"
    Set-List "@@LOF@@" "Figure"
    Set-List "@@LOT@@" "Table"

    # compact list entries so the contents fit one page: wdStyleTOC1 = -20, wdStyleTOC2 = -21, wdStyleTableOfFigures = -36
    foreach ($id in -20, -21, -36) {
        $pf = $doc.Styles.Item($id).ParagraphFormat
        $pf.SpaceBefore = 0
        $pf.SpaceAfter = 3
        $pf.LineSpacingRule = 0   # single
    }

    # fields everywhere: body, headers and footers; twice, since the lists shift the page numbers
    for ($pass = 0; $pass -lt 2; $pass++) {
        [void]$doc.Fields.Update()
        foreach ($s in $doc.Sections) {
            foreach ($i in 1..3) {
                [void]$s.Headers.Item($i).Range.Fields.Update()
                [void]$s.Footers.Item($i).Range.Fields.Update()
            }
        }
        $doc.Repaginate()
        foreach ($t in $doc.TablesOfContents) { [void]$t.Update() }
        foreach ($t in $doc.TablesOfFigures) { [void]$t.Update() }
    }

    $doc.Save()
    # wdExportFormatPDF = 17; OpenAfterExport false; wdExportOptimizeForPrint = 0; wdExportAllDocument = 0;
    # From/To unused; wdExportDocumentContent = 0; IncludeDocProps true; KeepIRM true; wdExportCreateHeadingBookmarks = 1
    $doc.ExportAsFixedFormat($Pdf, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1)
    Write-Output ("pages: " + $doc.ComputeStatistics(2))
    $doc.Close($false)
    Write-Output "saved $Docx"
    Write-Output "exported $Pdf"
} finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
