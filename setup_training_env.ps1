# setup_training_env.ps1
# ---------------------------------------------------------------------------
# One-click script to:
#   1. Download & install Python 3.12 silently
#   2. Create a virtual environment  (venv312)
#   3. Install PyTorch with CUDA 12.4 + JupyterLab + training deps
#   4. Register the kernel so VS Code / JupyterLab can use it
#   5. Verify CUDA is detected
# ---------------------------------------------------------------------------
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PROJECT  = Split-Path $MyInvocation.MyCommand.Path
$VENV_DIR = Join-Path $PROJECT "venv312"
$PY312    = "C:\Python312\python.exe"
$PY_DL    = "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe"
$PY_INST  = Join-Path $env:TEMP "python312_installer.exe"

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  DDoS Detector - GPU Training Environment Setup" -ForegroundColor Cyan
Write-Host "  NVIDIA RTX 3050 + CUDA 12.4 + Python 3.12" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# -- Step 1: Python 3.12 -----------------------------------------------------
if (Test-Path $PY312) {
    Write-Host "[1/4] Python 3.12 already installed at $PY312" -ForegroundColor Green
} else {
    Write-Host "[1/4] Python 3.12 not found - downloading installer ..." -ForegroundColor Yellow
    Write-Host "      URL : $PY_DL"
    Invoke-WebRequest -Uri $PY_DL -OutFile $PY_INST -UseBasicParsing
    Write-Host "      Installing silently (this takes ~1 min) ..."
    Start-Process -FilePath $PY_INST `
        -ArgumentList "/quiet InstallAllUsers=0 PrependPath=0 Include_pip=1 TargetDir=C:\Python312" `
        -Wait
    if (-not (Test-Path $PY312)) {
        Write-Error "Python 3.12 installation failed. Please install manually from https://www.python.org"
    }
    Write-Host "      Python 3.12 installed." -ForegroundColor Green
}

# -- Step 2: Virtual environment ---------------------------------------------
if (Test-Path $VENV_DIR) {
    Write-Host "[2/4] venv312 already exists - skipping creation." -ForegroundColor Green
} else {
    Write-Host "[2/4] Creating virtual environment venv312 ..." -ForegroundColor Yellow
    & $PY312 -m venv $VENV_DIR
    Write-Host "      Done." -ForegroundColor Green
}

$VENV_PY  = Join-Path $VENV_DIR "Scripts\python.exe"
$VENV_PIP = Join-Path $VENV_DIR "Scripts\pip.exe"

# -- Step 3: Install dependencies --------------------------------------------
Write-Host "[3/4] Installing PyTorch CUDA 12.4 + training dependencies ..." -ForegroundColor Yellow
Write-Host "      (PyTorch CUDA download is ~2.5 GB - please wait)"
& $VENV_PIP install --upgrade pip -q

# Install PyTorch with CUDA 12.4 support
& $VENV_PIP install torch --index-url https://download.pytorch.org/whl/cu124

# Install remaining training deps
& $VENV_PIP install numpy pandas scikit-learn joblib matplotlib seaborn jupyterlab ipywidgets ipykernel

Write-Host "      All packages installed." -ForegroundColor Green

# -- Step 4: Register Jupyter kernel -----------------------------------------
Write-Host "[4/4] Registering Jupyter kernel 'Python 3.12 (GPU)' ..." -ForegroundColor Yellow
& $VENV_PY -m ipykernel install --user --name=venv312 --display-name "Python 3.12 (GPU)"
Write-Host "      Kernel registered." -ForegroundColor Green

# -- Verify CUDA -------------------------------------------------------------
Write-Host ""
Write-Host "Verifying CUDA availability ..." -ForegroundColor Yellow
& $VENV_PY -c "import torch; cuda=torch.cuda.is_available(); gpu=torch.cuda.get_device_name(0) if cuda else 'NOT DETECTED'; print('  PyTorch :', torch.__version__); print('  CUDA    :', cuda); print('  GPU     :', gpu)"

# -- Done --------------------------------------------------------------------
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To start training:" -ForegroundColor White
Write-Host "    1. Launch Jupyter :  .\venv312\Scripts\jupyter.exe lab train_bilstm.ipynb" -ForegroundColor Gray
Write-Host "    2. Select kernel  :  Python 3.12 (GPU)  (top-right dropdown)" -ForegroundColor Gray
Write-Host "    3. Run all cells  :  Kernel > Restart Kernel and Run All Cells" -ForegroundColor Gray
Write-Host ""
Write-Host "  To start the web server (no venv needed):" -ForegroundColor White
Write-Host "    python -m uvicorn app:app --host 0.0.0.0 --port 8000" -ForegroundColor Gray
Write-Host ""
