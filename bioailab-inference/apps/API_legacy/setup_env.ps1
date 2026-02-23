# setup_env.ps1
$envName = "bioailab_ml_api"
$envPath = ".\.venv\$envName"

$packages = @(
    "numpy>=1.24",
    "pandas>=1.5",
    "scikit-learn>=1.2",
    "xgboost>=1.7",
    "matplotlib>=3.7",
    "PyYAML>=6.0",
    "joblib>=1.3",
    "fastapi>=0.110",
    "uvicorn>=0.23",
    "python-dotenv>=0.21.0"
)

function Ensure-Python {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "Python nao encontrado no PATH. Instale o Python 3.10+ e reinicie o terminal."
        exit 1
    }
}

function Ensure-Venv {
    if (-not (Test-Path $envPath)) {
        Write-Host "Criando ambiente virtual '$envName'..."
        python -m venv $envPath
    } else {
        Write-Host "Ambiente virtual '$envName' ja existe."
    }
}

function Activate-Venv {
    $activateScript = "$envPath\Scripts\Activate.ps1"
    if (-not (Test-Path $activateScript)) {
        Write-Error "Script de ativacao nao encontrado: $activateScript"
        exit 1
    }
    Write-Host "Ativando ambiente virtual '$envName'..."
    & $activateScript
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha ao ativar o ambiente virtual."
        exit 1
    }
}

function Install-Packages {
    Write-Host "Atualizando pip..."
    python -m pip install --upgrade pip

    foreach ($pkg in $packages) {
        $name = $pkg.Split(">=")[0]
        Write-Host "Verificando pacote '$name'..."
        python -m pip show $name > $null 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Instalando $pkg..."
            python -m pip install $pkg
        } else {
            Write-Host "$name ja esta instalado."
        }
    }
}

function Install-Torch {
    Write-Host "Detectando versao CUDA..."
    $cudaInfo = & nvidia-smi | Select-String "CUDA Version"
    if ($cudaInfo) {
        $cudaVersion = $cudaInfo -replace '.*CUDA Version:\s*([\d.]+).*', '$1'
        Write-Host "Versao CUDA detectada: $cudaVersion"
    } else {
        Write-Host "CUDA nao detectada; usando nightly/cu128."
        $cudaVersion = ""
    }

    switch -Wildcard ($cudaVersion) {
        "12.8*" { $url = "https://download.pytorch.org/whl/cu128" }
        "12.6*" { $url = "https://download.pytorch.org/whl/cu126" }
        "12.2*" { $url = "https://download.pytorch.org/whl/cu122" }
        default  { $url = "https://download.pytorch.org/whl/nightly/cu128" }
    }

    Write-Host "Verificando torch..."
    python -m pip show torch > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Instalando torch, torchvision e torchaudio de $url..."
        python -m pip install torch torchvision torchaudio --index-url $url
    } else {
        Write-Host "Atualizando torch..."
        python -m pip install --upgrade torch torchvision torchaudio --index-url $url
    }
}

# Main
Ensure-Python
Ensure-Venv
. "$envPath\Scripts\Activate.ps1"
Install-Packages
Install-Torch

python -m uvicorn api.api:app --reload

