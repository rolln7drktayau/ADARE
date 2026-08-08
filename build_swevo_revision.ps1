param(
    [switch]$SkipReports
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

if (-not $SkipReports) {
    Write-Host "Regenerating consolidated reports and publication assets..."
    python scripts/major_revision_report.py --root output/major_revision
    if ($LASTEXITCODE -ne 0) { throw "Report generation failed with exit code $LASTEXITCODE" }
    python scripts/build_revision_assets.py
    if ($LASTEXITCODE -ne 0) { throw "Publication asset generation failed with exit code $LASTEXITCODE" }
}

$pdfLatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$tectonic = Join-Path $projectRoot ".tools\tectonic-0.16.9\tectonic.exe"
if (-not $pdfLatex -and -not (Test-Path -LiteralPath $tectonic)) {
    throw "No LaTeX engine was found. Install MiKTeX or place Tectonic at $tectonic."
}

Write-Host "Compiling revised article and response letter..."
Push-Location -LiteralPath (Join-Path $projectRoot "papers")
try {
    foreach ($source in @("article_swevo.tex", "response_to_reviewers.tex")) {
        if ($pdfLatex) {
            1..2 | ForEach-Object {
                & $pdfLatex.Source -interaction=nonstopmode -halt-on-error $source
                if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed for $source" }
            }
        }
        else {
            & $tectonic $source --keep-logs --keep-intermediates
            if ($LASTEXITCODE -ne 0) { throw "Tectonic compilation failed for $source" }
        }
    }
}
finally {
    Pop-Location
}

$packageDir = Join-Path $projectRoot "submission\swevo_revision_2026"
$generatedDir = Join-Path $packageDir "generated"
$figureDir = Join-Path $packageDir "Figures"
$reportDir = Join-Path $packageDir "reports"
New-Item -ItemType Directory -Force -Path $packageDir,$generatedDir,$figureDir,$reportDir | Out-Null

Copy-Item -Force -LiteralPath `
    "papers\article_swevo.tex", `
    "papers\article_swevo.pdf", `
    "papers\response_to_reviewers.tex", `
    "papers\response_to_reviewers.pdf", `
    "papers\highlights.txt" `
    -Destination $packageDir
Copy-Item -Force -Path "papers\generated\*.tex" -Destination $generatedDir
Copy-Item -Force -LiteralPath `
    "Figures\revision_controller_behavior.png", `
    "Figures\revision_convergence_vs_evaluations.png", `
    "Figures\revision_convergence_vs_time.png", `
    "Figures\revision_memory_scaling.png", `
    "Figures\revision_resource_sensitivity.png" `
    -Destination $figureDir
Copy-Item -Force -LiteralPath `
    "output\major_revision\reports\major_revision_summary.md", `
    "output\major_revision\reports\major_revision_statistics.csv", `
    "output\major_revision\reports\controller_behavior.csv", `
    "output\major_revision\reports\runtime_breakdown.csv", `
    "output\major_revision\reports\reward_survival_correlation.csv", `
    "output\major_revision\reports\evaluation_budget.csv", `
    "output\major_revision\reports\reviewer_response_matrix.md" `
    -Destination $reportDir

$zipPath = Join-Path $projectRoot "submission\swevo_revision_2026.zip"
Compress-Archive -Force -Path (Join-Path $packageDir "*") -DestinationPath $zipPath

Write-Host "Revision package ready:"
Write-Host "  Article:  $packageDir\article_swevo.pdf"
Write-Host "  Response: $packageDir\response_to_reviewers.pdf"
Write-Host "  ZIP:      $zipPath"
