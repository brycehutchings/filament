[CmdletBinding()]
param(
    [switch]$Configure,
    [switch]$Build,
    [switch]$Install,
    [ValidateSet('Debug', 'Release')]
    [string]$BuildType = 'Debug'
)

$ErrorActionPreference = 'Stop'

if (-not ($Configure -or $Build -or $Install)) {
    throw 'Specify at least one of -Configure, -Build, or -Install.'
}

$root = $PSScriptRoot
$buildTypeName = $BuildType.ToLowerInvariant()
$windowsBuild = Join-Path $root "out\windows-$buildTypeName-x64"
$androidBuild = Join-Path $root "out\android-$buildTypeName-aarch64"
$windowsInstall = Join-Path $root "out\windows-$buildTypeName\filament"
$androidInstall = Join-Path $root "out\android-$buildTypeName\filament"
$androidToolchain = Join-Path $root 'build\toolchain-aarch64-linux-android.cmake'

function Invoke-CMake {
    param(
        [Parameter(Mandatory)]
        [string]$Platform,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    Write-Host "`n=== $Platform ==="
    & cmake @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "CMake failed for $Platform with exit code $LASTEXITCODE."
    }
}

function Initialize-MsvcEnvironment {
    if (Get-Command cl.exe -CommandType Application -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw 'Visual Studio Installer could not be found.'
    }

    $installation = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($installation)) {
        throw 'A Visual Studio installation with the C++ x64 tools could not be found.'
    }

    $devCmd = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
    $command = "call `"$devCmd`" -no_logo -arch=x64 >nul && set"
    $variables = & $env:ComSpec /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio environment initialization failed with exit code $LASTEXITCODE."
    }

    foreach ($variable in $variables) {
        $separator = $variable.IndexOf('=')
        if ($separator -gt 0) {
            [Environment]::SetEnvironmentVariable($variable.Substring(0, $separator),
                $variable.Substring($separator + 1), 'Process')
        }
    }

    if (-not (Get-Command cl.exe -CommandType Application -ErrorAction SilentlyContinue)) {
        throw 'MSVC cl.exe was not found after initializing the Visual Studio environment.'
    }
}

function Remove-IncompatibleWindowsCompilerCache {
    $cache = Join-Path $windowsBuild 'CMakeCache.txt'
    if (-not (Test-Path $cache)) {
        return
    }

    $compilerEntry = Get-Content $cache | Where-Object {
        $_.StartsWith('CMAKE_CXX_COMPILER:FILEPATH=')
    } | Select-Object -First 1
    if (-not $compilerEntry) {
        return
    }

    $compiler = $compilerEntry.Substring($compilerEntry.IndexOf('=') + 1)
    if ([IO.Path]::GetFileName($compiler) -ieq 'cl.exe') {
        return
    }

    Write-Host "Removing Windows CMake cache configured with '$compiler'."
    Remove-Item $cache -Force
    Remove-Item (Join-Path $windowsBuild 'CMakeFiles') -Recurse -Force -ErrorAction SilentlyContinue
}

Initialize-MsvcEnvironment

if ($Configure) {
    Remove-IncompatibleWindowsCompilerCache
    Invoke-CMake 'Configuring Windows' @(
        '-S', $root,
        '-B', $windowsBuild,
        '-G', 'Ninja',
        '-DCMAKE_C_COMPILER=cl.exe',
        '-DCMAKE_CXX_COMPILER=cl.exe',
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_INSTALL_PREFIX=$windowsInstall",
        '-DFILAMENT_BUILD_TESTING=OFF',
        '-DFILAMENT_ENABLE_OPENXR=ON',
        '-DFILAMENT_SUPPORTS_VULKAN=ON',
        '-DFILAMENT_SAMPLES_STEREO_TYPE=multiview'
    )

    Invoke-CMake 'Configuring Android arm64' @(
        '-S', $root,
        '-B', $androidBuild,
        '-G', 'Ninja',
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_TOOLCHAIN_FILE=$androidToolchain",
        "-DCMAKE_INSTALL_PREFIX=$androidInstall",
        '-DFILAMENT_BUILD_TESTING=OFF',
        '-DFILAMENT_ENABLE_OPENXR=OFF',
        '-DFILAMENT_SUPPORTS_VULKAN=ON',
        '-DFILAMENT_SAMPLES_STEREO_TYPE=multiview',
        '-DFILAMENT_SKIP_SAMPLES=ON',
        '-DFILAMENT_IMPORT_PREBUILT_EXECUTABLES_DIR=out'
    )
}

if ($Build) {
    Invoke-CMake 'Building Windows' @('--build', $windowsBuild, '--config', $BuildType)
    Invoke-CMake 'Building Android arm64' @('--build', $androidBuild, '--config', $BuildType)
}

if ($Install) {
    Invoke-CMake 'Installing Windows' @('--install', $windowsBuild, '--config', $BuildType)
    Invoke-CMake 'Installing Android arm64' @('--install', $androidBuild, '--config', $BuildType)
}