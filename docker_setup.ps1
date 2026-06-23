# ─────────────────────────────────────────────────────────────────
#  docker_setup.ps1 — Перший запуск проєкту в Docker (Windows)
#
#  Запуск у PowerShell:
#    .\docker_setup.ps1
# ─────────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"

function info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function step($msg)  { Write-Host "`n$msg" -ForegroundColor White }

step "=== Home Credit Default Risk — Docker Setup ==="

# ── 1. .env ────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    warn ".env не знайдено — створено з .env.example."
}

# ── 2. Directories ─────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path `
    "models\plots", "data\parquet", `
    "data\home-credit-default-risk", "docker\initdb" | Out-Null
info "Директорії створено."

# ── 3. Build & start ───────────────────────────────────────────
step "Крок 1/4 — Збірка Docker образу..."
docker compose build

step "Крок 2/4 — Запуск PostgreSQL..."
docker compose up -d db
info "Чекаємо 5 секунд на PostgreSQL..."
Start-Sleep -Seconds 5

# ── 4. Import data ─────────────────────────────────────────────
$csvCount = (Get-ChildItem "data\home-credit-default-risk\*.csv" -ErrorAction SilentlyContinue).Count
if ($csvCount -gt 0) {
    step "Крок 3/4 — Імпорт CSV → PostgreSQL ($csvCount файлів)..."
    docker compose run --rm app python init_db.py
    info "Дамп PostgreSQL → parquet..."
    docker compose run --rm app python dump_to_parquet.py
} else {
    warn "Крок 3/4 — CSV файли не знайдено."
    warn "  Помістіть їх у data\home-credit-default-risk\ та виконайте:"
    warn "    docker compose run --rm app python init_db.py"
    warn "    docker compose run --rm app python dump_to_parquet.py"
}

# ── 5. Start app ───────────────────────────────────────────────
step "Крок 4/4 — Запуск Streamlit..."
docker compose up -d app

info "Готово! Відкрийте: http://localhost:8501"
