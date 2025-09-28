$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

$python = "python"
$cmd = @($python, "-m", "src.ratings.pipeline", "preview")
Write-Output ("Running: " + ($cmd -join ' '))
& $python -m src.ratings.pipeline preview
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
