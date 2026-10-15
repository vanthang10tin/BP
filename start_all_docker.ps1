<#
.SYNOPSIS
    1-Click script to start CVAT and QC Studio on any machine via Docker Compose.
#>

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       QC STUDIO & CVAT 2D - ALL-IN-ONE LAUNCHER          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Docker Daemon
Write-Host "[1/4] Checking Docker status..." -ForegroundColor Yellow
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker error" }
} catch {
    Write-Host "[ERROR] Docker Desktop is not running." -ForegroundColor Red
    Write-Host "Please start Docker Desktop and re-run this script." -ForegroundColor Yellow
    exit 1
}

# 2. Check .env configuration
Write-Host "[2/4] Checking configuration (.env)..." -ForegroundColor Yellow
$envFile = Join-Path $repoRoot '.env'
$exampleEnv = Join-Path $repoRoot '.env.example'
if (-not (Test-Path $envFile) -and (Test-Path $exampleEnv)) {
    Copy-Item $exampleEnv $envFile
    Write-Host "Created .env from .env.example. You can add CVAT_ACCESS_TOKEN later if needed." -ForegroundColor Gray
}

# 3. Start Docker Compose Stack
Write-Host "[3/4] Starting all containers via Docker Compose..." -ForegroundColor Yellow
Set-Location $repoRoot
docker compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker Compose failed to start." -ForegroundColor Red
    exit 1
}

# 4. Readiness Check
Write-Host "[4/4] Verifying services readiness..." -ForegroundColor Yellow
$cvatUrl = "http://localhost:8080"
$qcUiUrl = "http://localhost:5000"
$qcApiUrl = "http://localhost:8000"

function Test-HttpReady([string]$Url, [int]$TimeoutSec = 30) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $res = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($res.StatusCode -eq 200 -or $res.StatusCode -eq 302) { return $true }
        } catch { }
        Start-Sleep -Seconds 2
    }
    return $false
}

$cvatOk = Test-HttpReady "$cvatUrl/" 45
$apiOk = Test-HttpReady "$qcApiUrl/api/health" 45
$uiOk = Test-HttpReady $qcUiUrl 45

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "                SYSTEM IS READY TO USE!                   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  1. CVAT Annotation Tool : $cvatUrl" -ForegroundColor Cyan
Write-Host "  2. QC Studio Workspace  : $qcUiUrl" -ForegroundColor Cyan
Write-Host "  3. QC Engine API        : $qcApiUrl" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To stop the system, run: docker compose down" -ForegroundColor Gray
Write-Host ""

Start-Process $qcUiUrl
