# PowerShell Script: Launch System & Expose via Free Public HTTPS URLs
$ErrorActionPreference = "Stop"

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🚀 Launching Local RAG System (FastAPI + Streamlit)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

$ProjectRoot = $PSScriptRoot
Set-Location -Path $ProjectRoot

$env:HF_HOME = "$ProjectRoot\models\huggingface"
$env:TRANSFORMERS_CACHE = "$ProjectRoot\models\huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$PythonPath = "$ProjectRoot\venv\Scripts\python.exe"

# 1. Start FastAPI Backend
Write-Host ">>> 1. Starting FastAPI Backend (Port 8000)..." -ForegroundColor Green
$BackendJob = Start-Job -ScriptBlock {
    param($ProjRoot, $Py)
    Set-Location -Path $ProjRoot
    $env:HF_HOME = "$ProjRoot\models\huggingface"
    $env:TRANSFORMERS_CACHE = "$ProjRoot\models\huggingface"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    & $Py -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
} -ArgumentList $ProjectRoot, $PythonPath

Start-Sleep -Seconds 3

# 2. Start Streamlit Frontend
Write-Host ">>> 2. Starting Streamlit Frontend (Port 8501)..." -ForegroundColor Green
$FrontendJob = Start-Job -ScriptBlock {
    param($ProjRoot, $Py)
    Set-Location -Path $ProjRoot
    & $Py -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1 --server.enableCORS false
} -ArgumentList $ProjectRoot, $PythonPath

Start-Sleep -Seconds 3

# Find Local IP Address
$LocalIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi","Ethernet*" -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike "127.*" } | Select-Object -First 1).IPAddress
if (-not $LocalIP) { $LocalIP = "127.0.0.1" }

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🌐 LOCAL & NETWORK ACCESS URLS:" -ForegroundColor Cyan
Write-Host "  * Local Machine UI : http://localhost:8501" -ForegroundColor Yellow
Write-Host "  * Local Wi-Fi Network UI : http://${LocalIP}:8501" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host ">>> Generating Free Public Web Tunnel (Localtunnel)..." -ForegroundColor Yellow

# Launch localtunnel via npx
npx -y localtunnel --port 8501
