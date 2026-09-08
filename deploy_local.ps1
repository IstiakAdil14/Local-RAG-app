# PowerShell Local Deployment Script for Local RAG System
$ErrorActionPreference = "Stop"

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🚀 Launching Local RAG System (Local Deployment)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

$ProjectRoot = $PSScriptRoot
Set-Location -Path $ProjectRoot

# Environment Configuration
$env:HF_HOME = "$ProjectRoot\models\huggingface"
$env:TRANSFORMERS_CACHE = "$ProjectRoot\models\huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Ensure Python Virtual Environment exists
$PythonPath = "$ProjectRoot\venv\Scripts\python.exe"
if (-not (Test-Path $PythonPath)) {
    Write-Host "❌ Virtual environment not found at $PythonPath." -ForegroundColor Red
    Write-Host "Please create venv first: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

Write-Host ">>> Starting FastAPI Backend on http://0.0.0.0:8000..." -ForegroundColor Green
$BackendJob = Start-Job -ScriptBlock {
    param($ProjRoot, $Py)
    Set-Location -Path $ProjRoot
    $env:HF_HOME = "$ProjRoot\models\huggingface"
    $env:TRANSFORMERS_CACHE = "$ProjRoot\models\huggingface"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    & $Py -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
} -ArgumentList $ProjectRoot, $PythonPath

Write-Host ">>> Waiting for FastAPI server to become active..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

try {
    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -Method Get
    if ($Health.status -eq "online") {
        Write-Host "✅ FastAPI Backend is Online!" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Backend is still initializing, proceeding to launch frontend..." -ForegroundColor Yellow
}

# Find local IP address
$LocalIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi","Ethernet*" -ErrorAction SilentlyContinue | Where-Host -FilterScript { $_.IPAddress -notlike "127.*" } | Select-Object -First 1).IPAddress
if (-not $LocalIP) { $LocalIP = "127.0.0.1" }

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🌐 SYSTEM ACCESS URLS:" -ForegroundColor Cyan
Write-Host "  * Local Machine UI : http://localhost:8501" -ForegroundColor Yellow
Write-Host "  * Local Network UI : http://${LocalIP}:8501" -ForegroundColor Yellow
Write-Host "  * REST API Docs    : http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host ">>> Launching Streamlit Web UI on http://0.0.0.0:8501..." -ForegroundColor Green
& $PythonPath -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
