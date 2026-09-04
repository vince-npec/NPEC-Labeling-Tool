[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11",
    [switch]$OneFile,
    [string]$AppVersion = "2026.8.2.4",
    [ValidateSet("none", "yang", "all")]
    [string]$ModelProfile = "none",
    [switch]$SkipTensorFlowCollection
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Invoke-Checked([string]$Exe, [object[]]$ArgList, [string]$ErrorMessage) {
    $flatArgs = @()
    foreach ($arg in $ArgList) {
        if ($null -eq $arg) {
            continue
        }
        if ($arg -is [System.Array]) {
            foreach ($nested in $arg) {
                if ($null -ne $nested) {
                    $flatArgs += [string]$nested
                }
            }
        } else {
            $flatArgs += [string]$arg
        }
    }
    & $Exe @flatArgs
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Test-PythonInvocation([string]$Exe, [string[]]$Prefix) {
    try {
        & $Exe @($Prefix + @("-c", "import sys; print(sys.version)")) *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Get-PythonVersion([string]$Exe, [string[]]$Prefix) {
    $raw = & $Exe @($Prefix + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"))
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query Python version from '$Exe'."
    }
    return [string]::Join("", $raw).Trim()
}

function Resolve-PythonCommand([string]$PreferredVersion) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        if (Test-PythonInvocation -Exe "py" -Prefix @("-$PreferredVersion")) {
            try {
                $resolvedExe = (& py "-$PreferredVersion" "-c" "import sys; print(sys.executable)" | Out-String).Trim()
                if (-not [string]::IsNullOrWhiteSpace($resolvedExe) -and (Test-Path $resolvedExe)) {
                    return @{ Exe = $resolvedExe; Prefix = @() }
                }
            } catch {
                # Fallback to launcher form if executable resolution fails.
            }
            return @{ Exe = "py"; Prefix = @("-$PreferredVersion") }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pyExe = (Get-Command python).Source
        if (Test-PythonInvocation -Exe $pyExe -Prefix @()) {
            $ver = Get-PythonVersion -Exe $pyExe -Prefix @()
            if ($ver -eq $PreferredVersion) {
                return @{ Exe = $pyExe; Prefix = @() }
            }
        }
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        $pyExe = (Get-Command python3).Source
        if (Test-PythonInvocation -Exe $pyExe -Prefix @()) {
            $ver = Get-PythonVersion -Exe $pyExe -Prefix @()
            if ($ver -eq $PreferredVersion) {
                return @{ Exe = $pyExe; Prefix = @() }
            }
        }
    }

    $detected = ""
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $detected = (& py -0p 2>$null | Out-String).Trim()
        } catch {
            $detected = ""
        }
    }
    $detectedBlock = if ([string]::IsNullOrWhiteSpace($detected)) { "(none via py launcher)" } else { $detected }
    throw @"
Required Python runtime not found: $PreferredVersion

Install Python $PreferredVersion (64-bit), then rerun:
  build_windows.cmd

Detected Python runtimes:
$detectedBlock
"@
}

function Ensure-PipReady([string]$PythonExe) {
    # Some Windows/Conda setups fail on pip self-upgrade. Keep build moving as long as pip works.
    try {
        Invoke-Checked -Exe $PythonExe -ArgList @("-m", "pip", "install", "--upgrade", "pip") -ErrorMessage "pip upgrade failed."
    } catch {
        Write-Warning "pip self-upgrade failed in this environment. Continuing with bundled pip."
    }
    Invoke-Checked -Exe $PythonExe -ArgList @("-m", "pip", "install", "--upgrade", "setuptools", "wheel") -ErrorMessage "setuptools/wheel upgrade failed."
}

function Resolve-BuildPath([string]$DesiredPath, [string]$Label) {
    if (-not (Test-Path $DesiredPath)) {
        return $DesiredPath
    }

    try {
        Remove-Item -Recurse -Force -ErrorAction Stop $DesiredPath
        return $DesiredPath
    } catch {
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $fallbackPath = "$DesiredPath-$timestamp"
        Write-Warning "$Label path is locked or in use ('$DesiredPath'). Using fallback path '$fallbackPath' for this build."
        return $fallbackPath
    }
}

function Resolve-ValidatedPythonCommand([string]$PreferredVersion) {
    if ($PreferredVersion -notin @("3.11", "3.12")) {
        throw "Unsupported -PythonVersion '$PreferredVersion'. Use 3.11 or 3.12."
    }
    $pythonCommand = Resolve-PythonCommand -PreferredVersion $PreferredVersion
    $resolvedVersion = Get-PythonVersion -Exe $pythonCommand.Exe -Prefix $pythonCommand.Prefix
    if ($resolvedVersion -ne $PreferredVersion) {
        throw @"
Python mismatch.
Requested: $PreferredVersion
Resolved:  $resolvedVersion

Please install Python $PreferredVersion and rerun.
"@
    }
    return $pythonCommand
}

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
if ($env:CONDA_PREFIX) {
    Write-Warning "Conda environment detected ($env:CONDA_PREFIX). This build script uses its own venv, but Python resolution must still point to 3.11/3.12."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")
$buildVenv = Join-Path $scriptDir ".build-venv-win"
$distDir = Join-Path $scriptDir "dist-win"
$buildDir = Join-Path $scriptDir "build-win"
$appName = "NPEC Labeling Tool"

$pythonCommand = Resolve-ValidatedPythonCommand -PreferredVersion $PythonVersion
$resolvedVersion = Get-PythonVersion -Exe $pythonCommand.Exe -Prefix $pythonCommand.Prefix
Write-Host "Using Python runtime: $resolvedVersion"

Write-Step "[1/6] Creating Windows build virtual environment..."
$buildVenv = Resolve-BuildPath -DesiredPath $buildVenv -Label "Build virtual environment"
$distDir = Resolve-BuildPath -DesiredPath $distDir -Label "Windows dist"
$buildDir = Resolve-BuildPath -DesiredPath $buildDir -Label "Windows work"

Invoke-Checked -Exe $pythonCommand.Exe -ArgList ($pythonCommand.Prefix + @("-m", "venv", $buildVenv)) -ErrorMessage "Virtual environment creation failed."
if (-not (Test-Path (Join-Path $buildVenv "Scripts\python.exe"))) {
    throw "Virtual environment creation failed at '$buildVenv'. Check your Python installation and rerun."
}

$pipExe = Join-Path $buildVenv "Scripts\pip.exe"
$venvPythonExe = Join-Path $buildVenv "Scripts\python.exe"
$pyInstallerExe = Join-Path $buildVenv "Scripts\pyinstaller.exe"
$venvVersion = Get-PythonVersion -Exe $venvPythonExe -Prefix @()
if ($venvVersion -ne $PythonVersion) {
    throw "Build venv Python version mismatch. Expected $PythonVersion but got $venvVersion at $venvPythonExe."
}

Write-Step "[2/6] Installing dependencies..."
Ensure-PipReady -PythonExe $venvPythonExe
Invoke-Checked -Exe $venvPythonExe -ArgList @("-m", "pip", "install", "--only-binary=:all:", "-r", (Join-Path $scriptDir "requirements.txt")) -ErrorMessage "Dependency installation failed. Ensure Python 3.11/3.12 is used."
if (-not (Test-Path $pyInstallerExe)) {
    throw "PyInstaller was not installed in the build venv. Check dependency installation output."
}

$iconScriptPath = Join-Path $scriptDir "scripts\generate_app_icons.py"
$logoForIconPath = Join-Path $scriptDir "assets\npec-logo-label.jpg"
if ((Test-Path $iconScriptPath) -and (Test-Path $logoForIconPath)) {
    Write-Host "Generating app icon assets..." -ForegroundColor DarkGray
    try {
        Invoke-Checked -Exe $venvPythonExe -ArgList @($iconScriptPath, "--input", $logoForIconPath, "--output-dir", (Join-Path $scriptDir "assets")) -ErrorMessage "App icon generation failed."
    } catch {
        Write-Warning "Icon generation failed; continuing with existing icon assets."
    }
}

Write-Step "[3/6] Running PyInstaller for Windows..."
$pyInstallerArgs = @(
    "--noconfirm",
    "--windowed",
    "--name", $appName,
    "--clean",
    "--distpath", $distDir,
    "--workpath", $buildDir,
    "--specpath", $scriptDir,
    "--paths", $rootDir,
    (Join-Path $scriptDir "main.py")
)

$logoPath = Join-Path $scriptDir "assets\npec-logo-label.jpg"
if (Test-Path $logoPath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources\assets" -f $logoPath))
}

$appIconPngPath = Join-Path $scriptDir "assets\npec-app-icon.png"
if (Test-Path $appIconPngPath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources\assets" -f $appIconPngPath))
}

$rightDockLogoPath = Join-Path $scriptDir "assets\NPEC-logo-black.png"
if (Test-Path $rightDockLogoPath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources\assets" -f $rightDockLogoPath))
}

$startupSplashPath = Join-Path $scriptDir "assets\npec-label.jpeg"
if (Test-Path $startupSplashPath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources\assets" -f $startupSplashPath))
}
$foundationRunnerPath = Join-Path $scriptDir "foundation_external_runner.py"
if (Test-Path $foundationRunnerPath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources" -f $foundationRunnerPath))
}

$plantHealthRunnerPath = Join-Path $scriptDir "plant_health_runner.py"
if (Test-Path $plantHealthRunnerPath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources" -f $plantHealthRunnerPath))
}

$pyphenotyperPipelinePath = Join-Path $scriptDir "npec_pyphenotyper"
if (Test-Path $pyphenotyperPipelinePath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources\npec_pyphenotyper" -f $pyphenotyperPipelinePath))
}

$generalRootStarterPath = Join-Path $scriptDir "general_root_starter"
if (Test-Path $generalRootStarterPath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources\general_root_starter" -f $generalRootStarterPath))
}

$builtinModelRoot = Join-Path $scriptDir "builtin_models"
if ($ModelProfile -eq "none") {
    Write-Host "Building without bundled checkpoint binaries." -ForegroundColor DarkGray
} elseif ($ModelProfile -eq "yang") {
    $yangModelSpecs = @(
        @{
            Relative = "hades_lucifer\best_root_model_patch_256_max_f10.8_max_IoU0.905.h5"
            Sha256 = "2034ddeb36a980e2ebdb67871c059c2746ed7fbaba25e4e500d61bcb5a14059e"
        },
        @{
            Relative = "rgb_inoculated\rgb_inoculated_shoot_v3.keras"
            Sha256 = "5064497983311af53bc5a070f165fcf8ccb909226e42100d753c3ff059c962ea"
        },
        @{
            Relative = "rgb_inoculated\rgb_inoculated_shoot_v3.profile.json"
            Sha256 = "db54b5bf54ecccc0e79674351cf61dd00f7c8746abdf02ee1ad24f1b4931fe83"
        },
        @{
            Relative = "five_seedling_ownership\yang_rgb_owner_ranker_v1.json"
            Sha256 = "a000451852dfd3ea088abeddcf62799e495efd67533a6ac9032890f4779bcb85"
        }
    )
    foreach ($modelSpec in $yangModelSpecs) {
        $modelPath = Join-Path $builtinModelRoot $modelSpec.Relative
        if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
            throw "Yang Song workflow model profile is missing: $modelPath"
        }
        $actualSha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $modelSpec.Sha256) {
            throw "Yang Song workflow model checksum mismatch for '$($modelSpec.Relative)'. Expected $($modelSpec.Sha256), got $actualSha256."
        }
        $modelFamily = ($modelSpec.Relative -split "[\\/]")[0]
        $destination = "resources\builtin_models\$modelFamily"
        $pyInstallerArgs += @("--add-data", ("{0};{1}" -f $modelPath, $destination))
    }
} else {
    foreach ($modelFamily in @(
        "hades_lucifer",
        "potato",
        "mxlab_dark_rgb",
        "rgb_inoculated",
        "five_seedling_ownership",
        "general_root_starter_unified",
        "plant_health"
    )) {
        $modelPath = Join-Path $builtinModelRoot $modelFamily
        if (Test-Path -LiteralPath $modelPath -PathType Container) {
            $destination = "resources\builtin_models\$modelFamily"
            $pyInstallerArgs += @("--add-data", ("{0};{1}" -f $modelPath, $destination))
        }
    }
}

foreach ($legalFileName in @("LICENSE", "CITATION.cff", "AUTHORS.md", "THIRD_PARTY_NOTICES.md", "TRADEMARKS.md")) {
    $legalFilePath = Join-Path $rootDir $legalFileName
    if (Test-Path $legalFilePath) {
        $pyInstallerArgs += @("--add-data", ("{0};legal" -f $legalFilePath))
    }
}
$licenseDirectory = Join-Path $rootDir "licenses"
if (Test-Path -LiteralPath $licenseDirectory -PathType Container) {
    $pyInstallerArgs += @("--add-data", ("{0};legal\licenses" -f $licenseDirectory))
}

$appIconIcoPath = Join-Path $scriptDir "assets\npec-app-icon.ico"
if (Test-Path $appIconIcoPath) {
    $pyInstallerArgs += @("--icon", $appIconIcoPath)
}

$guidePath = Join-Path $scriptDir "output\pdf\NPEC_Labeling_Tool_Application_Guide.pdf"
if (Test-Path $guidePath) {
    $pyInstallerArgs += @("--add-data", ("{0};resources\output\pdf" -f $guidePath))
}

# The app loads parts of pyphenotyper dynamically at runtime, so these modules
# are not visible to static analysis and must be collected explicitly.
$runtimeCollectionArgs = @(
    "--hidden-import", "cv2",
    "--hidden-import", "imageio",
    "--hidden-import", "imageio_ffmpeg",
    "--hidden-import", "patchify",
    "--hidden-import", "skimage",
    "--hidden-import", "pandas",
    "--hidden-import", "rich",
    "--hidden-import", "rich.progress",
    "--hidden-import", "zarr",
    "--hidden-import", "keras",
    "--hidden-import", "keras.backend",
    "--hidden-import", "keras.models",
    "--hidden-import", "keras.layers",
    "--hidden-import", "keras.callbacks",
    "--collect-all", "cv2",
    "--collect-all", "imageio",
    "--collect-all", "imageio_ffmpeg",
    "--copy-metadata", "imageio",
    "--copy-metadata", "imageio_ffmpeg",
    "--collect-all", "patchify",
    "--collect-all", "skimage",
    "--copy-metadata", "scikit-image",
    "--hidden-import", "scipy",
    "--hidden-import", "scipy.spatial",
    "--hidden-import", "scipy.stats",
    "--collect-all", "scipy",
    "--copy-metadata", "scipy",
    "--collect-all", "pandas",
    "--collect-all", "rich",
    "--collect-all", "zarr",
    "--copy-metadata", "zarr",
    "--collect-all", "keras"
)

if (-not $SkipTensorFlowCollection) {
    $runtimeCollectionArgs += @(
        "--hidden-import", "tensorflow",
        "--hidden-import", "tensorflow.keras",
        "--hidden-import", "tensorflow.lite",
        "--hidden-import", "tensorflow.lite.python.lite",
        "--hidden-import", "tensorflow.lite.python.convert",
        "--hidden-import", "tensorflow.lite.python.util",
        "--hidden-import", "tensorflow.lite.python.interpreter",
        "--hidden-import", "tensorflow.lite.python.schema_py_generated",
        "--hidden-import", "tensorflow.lite.python.authoring.authoring",
        "--hidden-import", "h5py",
        "--collect-all", "tensorflow",
        "--collect-all", "h5py"
    )
    Write-Host "TensorFlow collection enabled (default)." -ForegroundColor DarkGray
} else {
    Write-Host "TensorFlow collection skipped by request (-SkipTensorFlowCollection)." -ForegroundColor DarkGray
}
$pyInstallerArgs += $runtimeCollectionArgs
if ($OneFile) {
    $pyInstallerArgs = @("--onefile") + $pyInstallerArgs
}
Invoke-Checked -Exe $pyInstallerExe -ArgList $pyInstallerArgs -ErrorMessage "PyInstaller build failed."

$oneDirPath = Join-Path $distDir $appName
$oneDirExePath = Join-Path $oneDirPath "$appName.exe"
$oneFileExePath = Join-Path $distDir "$appName.exe"
$builtExe = $null

if (Test-Path $oneDirExePath) {
    $builtExe = $oneDirExePath
} elseif (Test-Path $oneFileExePath) {
    $builtExe = $oneFileExePath
} else {
    throw "PyInstaller completed, but no executable was found."
}
Write-Host "Built executable: $builtExe"

Write-Step "[4/6] Creating ZIP package..."
$zipPath = Join-Path $distDir "NPEC-Labeling-Tool-windows.zip"
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
if (Test-Path $oneDirPath) {
    Compress-Archive -Path $oneDirPath -DestinationPath $zipPath -CompressionLevel Optimal
} else {
    Compress-Archive -Path $builtExe -DestinationPath $zipPath -CompressionLevel Optimal
}
Write-Host "Built ZIP: $zipPath"
$zipHash = Get-FileHash -Algorithm SHA256 $zipPath
Set-Content -Path "$zipPath.sha256" -Value (("{0}  {1}" -f $zipHash.Hash.ToLowerInvariant(), (Split-Path $zipPath -Leaf))) -Encoding ascii

Write-Step "[5/6] Attempting installer build with Inno Setup (optional)..."
$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if ($null -ne $iscc -and (Test-Path $oneDirPath)) {
    $issPath = Join-Path $scriptDir "windows_installer.iss"
    if (-not (Test-Path $issPath)) {
        Write-Warning "Inno Setup was found, but windows_installer.iss is missing. Skipping installer build."
    } else {
        & $iscc.Source "/DMyAppVersion=$AppVersion" "/DMySourceDir=$oneDirPath" "/DMyOutputDir=$distDir" $issPath
        $installerPath = Join-Path $distDir "NPEC-Labeling-Tool-Setup.exe"
        if (Test-Path $installerPath) {
            Write-Host "Built installer: $installerPath"
            $installerHash = Get-FileHash -Algorithm SHA256 $installerPath
            Set-Content -Path "$installerPath.sha256" -Value (("{0}  {1}" -f $installerHash.Hash.ToLowerInvariant(), (Split-Path $installerPath -Leaf))) -Encoding ascii
        } else {
            Write-Warning "Inno Setup finished, but no installer was found in $distDir."
        }
    }
} else {
    Write-Host "Inno Setup not found (or one-directory output missing). Skipping installer build."
}

Write-Step "[6/6] Done."
