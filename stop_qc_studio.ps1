$ErrorActionPreference = 'Stop'
$runtimeDir = Join-Path $PSScriptRoot '.qc-data/runtime'
foreach ($name in @('frontend', 'backend')) {
    $pidFile = Join-Path $runtimeDir "$name.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) { continue }
    $processId = [int](Get-Content -LiteralPath $pidFile -Raw)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($process) {
        $commandLine = [string]$process.CommandLine
        $isExpected = if ($name -eq 'backend') { $commandLine -match 'backend[\\/]app\.py' } else { $commandLine -match 'BlazorApp\.(csproj|dll|exe)' }
        if (-not $isExpected) {
            Write-Warning "Removing stale PID $processId from $name.pid because its command line does not match QC Studio."
            Remove-Item -LiteralPath $pidFile -Force
            continue
        }
        $children = if ($name -eq 'backend') {
            @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $processId" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'backend[\\/]app\.py' })
        } else {
            @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $processId" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'BlazorApp\.(dll|exe)' })
        }
        foreach ($child in $children) {
            Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped $name child (PID $($child.ProcessId))."
        }
        Stop-Process -Id $processId -Force
        Write-Host "Stopped $name (PID $processId)."
    }
    Remove-Item -LiteralPath $pidFile -Force
}
$frontendExe = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'frontend/BlazorApp/bin/Release/net10.0/BlazorApp.exe'))
$orphanedFrontends = Get-CimInstance Win32_Process -Filter "Name = 'BlazorApp.exe'" -ErrorAction SilentlyContinue | Where-Object {
    $_.ExecutablePath -and [System.IO.Path]::GetFullPath($_.ExecutablePath) -eq $frontendExe
}
foreach ($process in $orphanedFrontends) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped orphaned frontend (PID $($process.ProcessId))."
}
Write-Host 'QC Studio processes stopped successfully.'
