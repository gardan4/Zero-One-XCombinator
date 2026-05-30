# Start backend + frontend (Windows). Prefer: just dev  or  uv run python scripts/dev.py
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
uv run python (Join-Path $PSScriptRoot "dev.py")
