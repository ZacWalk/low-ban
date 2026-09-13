#requires -Version 7.4
# dd project command backing script for low-ban.
#
# Ported from the pre-dd bespoke driver. Scores the PNG samples in samples/
# offline through the codec, with no camera needed.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$request = [Console]::In.ReadLine() | ConvertFrom-Json -AsHashtable
$root = Split-Path $PSScriptRoot
if ($request.schema -ne 1 -or $request.projectRoot -ne $root -or $request.parameters -isnot [hashtable] -or
    $request.dryRun -isnot [bool] -or $request.command -notin @('evaluate')) {
    throw 'Invalid project-command request.'
}
Set-Location $root

$p = $request.parameters
$config = if ($p.config) { [string]$p.config } else { 'release' }
$suffix = if ($config -eq 'debug') { 'd' } else { '' }
$ext = if ($IsWindows) { '.exe' } else { '' }
$app = Join-Path $root "exe/low-ban-64$suffix$ext"

# The landmark model is optional: the codec runs without it, but the face-weighted
# importance map is what the evaluation is actually measuring.
$model = Join-Path $root 'exe/shape_predictor_68_face_landmarks.dat'
$samples = @(Get-ChildItem (Join-Path $root 'samples') -Filter '*.png' -File -ErrorAction SilentlyContinue)

$plan = @{ command = 'evaluate'; binary = $app; arguments = @('/evaluate')
    sampleCount = $samples.Count; landmarkModel = [bool](Test-Path -LiteralPath $model) }

if ($request.dryRun) {
    @{ schema = 1; data = @{ status = 'planned'; plan = $plan }; files = @() } |
        ConvertTo-Json -Depth 10 -Compress | Write-Output
    exit 0
}

if (-not (Test-Path -LiteralPath $app)) { throw "Application not built: $app. Run dd build $config first." }
if (-not $samples.Count) { throw 'No PNG samples found in samples/.' }

$output = & $app '/evaluate' 2>&1 | Out-String
$code = $LASTEXITCODE

@{ schema = 1; data = @{ status = 'complete'; plan = $plan; exitCode = $code; output = $output }; files = @() } |
    ConvertTo-Json -Depth 10 -Compress | Write-Output
exit $code
