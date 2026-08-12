<#
.SYNOPSIS
    Build, run and test helper for low-ban.

.DESCRIPTION
    Defaults to Debug, except for `run` which defaults to Release so the demo is shown at
    full speed. Pass -Config to override.

.EXAMPLE
    .\dd.ps1 build
    .\dd.ps1 run
    .\dd.ps1 run -Config Debug
    .\dd.ps1 test
    .\dd.ps1 clean
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('build', 'run', 'test', 'evaluate', 'clean')]
    [string]$Command = 'build',

    [ValidateSet('Debug', 'Release')]
    [string]$Config
)

$ErrorActionPreference = 'Stop'
if (-not $Config) { $Config = if ($Command -eq 'run') { 'Release' } else { 'Debug' } }

$root = $PSScriptRoot
$solution = Join-Path $root 'low-ban.sln'
$exeName = if ($Config -eq 'Debug') { 'low-ban-64d.exe' } else { 'low-ban-64.exe' }
$exe = Join-Path $root "exe\$exeName"

function Get-MSBuild {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $found = & $vswhere -latest -prerelease -products * `
            -requires Microsoft.Component.MSBuild `
            -find 'MSBuild\**\Bin\amd64\MSBuild.exe' | Select-Object -First 1
        if ($found) { return $found }
    }

    $fallback = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
    if ($fallback) { return $fallback.Source }

    throw 'MSBuild was not found. Install Visual Studio with the C++ workload.'
}

function Invoke-Build {
    $msbuild = Get-MSBuild
    Write-Host "building $Config|x64" -ForegroundColor Cyan
    & $msbuild $solution /nologo /v:minimal /m "/p:Configuration=$Config" '/p:Platform=x64'
    if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
}

function Assert-Model {
    $model = Join-Path $root 'exe\shape_predictor_68_face_landmarks.dat'
    if (-not (Test-Path $model)) {
        Write-Warning "exe\shape_predictor_68_face_landmarks.dat is missing; landmarking will be disabled."
    }
}

switch ($Command) {
    'build' {
        Invoke-Build
    }

    'run' {
        Invoke-Build
        Assert-Model
        Write-Host "running $exeName" -ForegroundColor Cyan
        Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe)
    }

    'test' {
        Invoke-Build
        Assert-Model
        Write-Host "running self test" -ForegroundColor Cyan
        $process = Start-Process -FilePath $exe -ArgumentList '/test' `
            -WorkingDirectory (Split-Path $exe) -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "self test failed ($($process.ExitCode))" }
        Write-Host 'self test passed' -ForegroundColor Green
    }

    'evaluate' {
        Invoke-Build
        Assert-Model
        Write-Host 'evaluating PNG samples' -ForegroundColor Cyan
        $process = Start-Process -FilePath $exe -ArgumentList '/evaluate' `
            -WorkingDirectory (Split-Path $exe) -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "evaluation failed ($($process.ExitCode))" }
    }

    'clean' {
        foreach ($path in @((Join-Path $root 'intermediate'), (Join-Path $root 'exe\low-ban-64.exe'),
                (Join-Path $root 'exe\low-ban-64d.exe'), (Join-Path $root 'exe\low-ban-64.pdb'),
                (Join-Path $root 'exe\low-ban-64d.pdb'))) {
            if (Test-Path $path) { Remove-Item $path -Recurse -Force }
        }
        Write-Host 'cleaned' -ForegroundColor Green
    }
}
