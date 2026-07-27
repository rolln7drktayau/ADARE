param(
    [ValidateSet("full", "quick", "reports")]
    [string]$Preset = "full",

    [switch]$PlanOnly,

    [string]$StartAt = "",

    [string[]]$Only = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = "python"
$OutputRoot = Join-Path $Root "output\major_revision"
$LogRoot = Join-Path $OutputRoot "logs"
$ReportRoot = Join-Path $OutputRoot "reports"
New-Item -ItemType Directory -Force $LogRoot, $ReportRoot | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = Join-Path $LogRoot "major_revision_${Preset}_${Stamp}_transcript.log"

Start-Transcript -Path $TranscriptPath | Out-Null
try {
    Write-Host "SwEvo major revision runner"
    Write-Host "Root: $Root"
    Write-Host "Preset: $Preset"
    Write-Host "Transcript: $TranscriptPath"
    Write-Host ""

    & $Python --version
    if ($LASTEXITCODE -ne 0) {
        throw "Python is not available from PATH."
    }

    $ArgsList = @("scripts/major_revision_pipeline.py", "--preset", $Preset)
    if ($StartAt.Trim().Length -gt 0) {
        $ArgsList += @("--start-at", $StartAt)
    }
    if ($Only.Count -gt 0) {
        $ArgsList += "--only"
        $ArgsList += $Only
    }
    if (-not $PlanOnly) {
        $ArgsList += "--execute"
    }

    Write-Host "Command:"
    Write-Host "$Python $($ArgsList -join ' ')"
    Write-Host ""

    & $Python @ArgsList
    $ExitCode = $LASTEXITCODE

    Write-Host ""
    Write-Host "Finished with exit code: $ExitCode"
    Write-Host "Reports:"
    Write-Host "  $ReportRoot\major_revision_summary.md"
    Write-Host "  $ReportRoot\major_revision_statistics.csv"
    Write-Host "  $ReportRoot\controller_behavior.csv"
    Write-Host "  $ReportRoot\reviewer_response_matrix.md"
    Write-Host ""
    Write-Host "Logs:"
    Write-Host "  $LogRoot"

    exit $ExitCode
}
finally {
    Stop-Transcript | Out-Null
}
