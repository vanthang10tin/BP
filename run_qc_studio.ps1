param(
    [switch]$NoBrowser,
    [switch]$SkipDependencyCheck,
    [switch]$Wait
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$appRoot = Join-Path $repoRoot 'source-tool/qc-bdd100k'
$frontendRoot = Join-Path $appRoot 'frontend/BlazorApp'
$runtimeDir = Join-Path $appRoot '.qc-data/runtime'
$backendPidFile = Join-Path $runtimeDir 'backend.pid'
$frontendPidFile = Join-Path $runtimeDir 'frontend.pid'
$backendLog = Join-Path $runtimeDir 'backend.log'
$backendErrorLog = Join-Path $runtimeDir 'backend-error.log'
$frontendLog = Join-Path $runtimeDir 'frontend.log'
$dotnetHome = Join-Path $repoRoot '.tools/dotnet-home'
$nugetPackages = Join-Path $repoRoot '.tools/nuget-packages'
$appData = Join-Path $repoRoot '.tools/appdata'
$apiUrl = 'http://127.0.0.1:8000'
$uiUrl = 'http://127.0.0.1:5000'

function Test-Http([string]$Url) {
    try { Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing | Out-Null; return $true }
    catch { return $false }
}

function Resolve-Dotnet {
    $local = Join-Path $repoRoot '.tools/dotnet/dotnet.exe'
    if (Test-Path -LiteralPath $local) { return $local }
    $command = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw 'Could not find .NET 10 SDK. Install it or place the portable SDK in .tools/dotnet.'
}

function Resolve-Python {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $resolved = (& $python.Source -c 'import sys; print(sys.executable)' 2>$null | Select-Object -First 1)
        if ($resolved -and (Test-Path -LiteralPath $resolved.Trim())) { return @{ File = $resolved.Trim(); Prefix = @() } }
        return @{ File = $python.Source; Prefix = @() }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $resolved = (& $py.Source -3 -c 'import sys; print(sys.executable)' 2>$null | Select-Object -First 1)
        if ($resolved -and (Test-Path -LiteralPath $resolved.Trim())) { return @{ File = $resolved.Trim(); Prefix = @() } }
        return @{ File = $py.Source; Prefix = @('-3') }
    }
    throw 'Could not find Python 3.10+. Install Python and add it to PATH.'
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
New-Item -ItemType Directory -Path $dotnetHome, $nugetPackages, (Join-Path $appData 'NuGet') -Force | Out-Null
$env:DOTNET_CLI_HOME = $dotnetHome
$env:NUGET_PACKAGES = $nugetPackages
$env:APPDATA = $appData
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = '1'
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$env:ASPNETCORE_ENVIRONMENT = 'Development'
$repoNugetConfig = Join-Path $repoRoot '.tools/NuGet.Config'
if (Test-Path -LiteralPath $repoNugetConfig) {
    Copy-Item -LiteralPath $repoNugetConfig -Destination (Join-Path $appData 'NuGet/NuGet.Config') -Force
}

$envPath = Join-Path $repoRoot '.env'
if (Test-Path -LiteralPath $envPath) {
    Get-Content -LiteralPath $envPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $parts = $line.Split('=', 2)
            $k = $parts[0].Trim()
            $v = $parts[1].Trim().Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($k, $v, 'Process')
        }
    }
}

if ((Test-Http "$apiUrl/api/health") -and (Test-Http $uiUrl)) {
    Write-Host "QC Studio is already running: $uiUrl" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $uiUrl }
    exit 0
}

$pythonInfo = Resolve-Python
$dotnet = Resolve-Dotnet
if (-not $SkipDependencyCheck) {
    & $pythonInfo.File @($pythonInfo.Prefix) -c 'import fastapi, uvicorn, PIL, numpy, shapely' 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'Missing Python packages. Run: python -m pip install -r source-tool/qc-bdd100k/requirements.txt'
    }
}

if (-not (Test-Http "$apiUrl/api/health")) {
    if (Test-Path -LiteralPath $backendPidFile) {
        $oldPid = [int](Get-Content -LiteralPath $backendPidFile -Raw)
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
        if ($proc) {
            $cmd = [string]$proc.CommandLine
            if ($cmd -match 'backend[\\/]app\.py') {
                throw "Backend PID $oldPid exists but the health check failed. Inspect $backendLog; refusing to kill an unrelated process."
            } else {
                Write-Warning "Removing stale backend PID file (PID $oldPid was reused by another process: $($proc.Name))."
            }
        }
        Remove-Item -LiteralPath $backendPidFile -Force
    }
    $backendArgs = @($pythonInfo.Prefix) + @('-u', 'backend/app.py')
    $emptyIn = Join-Path $runtimeDir 'empty.in'
    Set-Content -LiteralPath $emptyIn -Value '' -NoNewline
    $backend = Start-Process -FilePath $pythonInfo.File -ArgumentList $backendArgs -WorkingDirectory $appRoot -PassThru -RedirectStandardInput $emptyIn -RedirectStandardOutput $backendLog -RedirectStandardError $backendErrorLog -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(45)
    while ((Get-Date) -lt $deadline -and -not (Test-Http "$apiUrl/api/health")) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Http "$apiUrl/api/health")) { throw "Backend did not become ready in 45 seconds. Inspect log: $backendLog" }
    $listeningPid = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
    $actualPid = if ($listeningPid) { $listeningPid } else { $backend.Id }
    Set-Content -LiteralPath $backendPidFile -Value $actualPid -NoNewline
}

if (-not (Test-Http $uiUrl)) {
    if (Test-Path -LiteralPath $frontendPidFile) {
        $oldPid = [int](Get-Content -LiteralPath $frontendPidFile -Raw)
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
        if ($proc) {
            $cmd = [string]$proc.CommandLine
            if ($cmd -match 'BlazorApp\.(csproj|dll|exe)') {
                throw "Frontend PID $oldPid exists but the health check failed. Inspect $frontendLog; refusing to kill an unrelated process."
            } else {
                Write-Warning "Removing stale frontend PID file (PID $oldPid was reused by another process: $($proc.Name))."
            }
        }
        Remove-Item -LiteralPath $frontendPidFile -Force
    }
    $env:QcApi__BaseUrl = $apiUrl
    $projectFile = Join-Path $frontendRoot 'BlazorApp.csproj'
    if (Test-Path -LiteralPath $repoNugetConfig) {
        & $dotnet restore $projectFile --configfile $repoNugetConfig *> $frontendLog
    } else {
        & $dotnet restore $projectFile *> $frontendLog
    }
    if ($LASTEXITCODE -ne 0) { throw "Frontend package restore failed. Inspect log: $frontendLog" }
    & $dotnet build $projectFile --configuration Release --no-restore *>> $frontendLog
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed. Inspect log: $frontendLog" }
    # `dotnet run` configures the Static Web Assets manifest; invoking the DLL
    # directly serves HTML but misses CSS/JS assets in a build (non-publish) output.
    $emptyIn = Join-Path $runtimeDir 'empty.in'
    if (-not (Test-Path -LiteralPath $emptyIn)) { Set-Content -LiteralPath $emptyIn -Value '' -NoNewline }
    $frontend = Start-Process -FilePath $dotnet -ArgumentList @('run', '--no-build', '--no-launch-profile', '--project', 'BlazorApp.csproj', '--environment', 'Development', '--configuration', 'Release', '--urls', $uiUrl) -WorkingDirectory $frontendRoot -PassThru -RedirectStandardInput $emptyIn -RedirectStandardOutput (Join-Path $runtimeDir 'frontend-server.log') -RedirectStandardError (Join-Path $runtimeDir 'frontend-error.log') -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline -and -not (Test-Http $uiUrl)) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Http $uiUrl)) { throw "Frontend did not become ready in 90 seconds. Inspect log: $frontendLog" }
    $listeningPid = (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
    $actualPid = if ($listeningPid) { $listeningPid } else { $frontend.Id }
    Set-Content -LiteralPath $frontendPidFile -Value $actualPid -NoNewline
}

Write-Host "QC Studio is ready: $uiUrl" -ForegroundColor Green
Write-Host "API: $apiUrl | CVAT (if its gateway is running): http://127.0.0.1:8080"
Write-Host "Logs: $runtimeDir"
if (-not $NoBrowser) { Start-Process $uiUrl }

if ($Wait) {
    Write-Host "Services are running. Press Ctrl+C to stop."
    try {
        while ($true) { Start-Sleep -Seconds 3600 }
    } finally {
        $stopScript = Join-Path $repoRoot 'stop_qc_studio.ps1'
        if (Test-Path -LiteralPath $stopScript) { & $stopScript }
    }
}
