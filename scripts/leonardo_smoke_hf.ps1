param(
    [string]$Config = "packages/training/configs/leonardo_smoke_hf.yaml",
    [switch]$WaitAndUpload
)

$ErrorActionPreference = "Stop"

# Leonardo pipeline — prefer:  uv run python scripts/leonardo_smoke.py
# This PowerShell wrapper is kept for compatibility; same tar+ssh flow as the Python script.
# 1. Sync repo to the login node.
# 2. Pre-stage on login: uv/gpu deps, HF weights, import warm-up (leonardo_remote_prestage.sh).
# 3. SLURM GPU job: offline HF, live W&B via proxy (XCombinator/XCombinator).
# 4. After job: upload adapter to Hugging Face from login (-WaitAndUpload).

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvPath = Join-Path $Root ".env"

if (-not (Test-Path $EnvPath)) {
    throw "Missing .env. Copy .env.example to .env and fill in the cluster/Hugging Face values."
}

Get-Content $EnvPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
        return
    }
    $key, $value = $line.Split("=", 2)
    [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), "Process")
}

foreach ($name in @("ZO_CLUSTER_HOST", "ZO_CLUSTER_USER", "ZO_CLUSTER_REPO_DIR", "HF_TOKEN")) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        throw "Set $name in .env"
    }
}

$target = "$env:ZO_CLUSTER_USER@$env:ZO_CLUSTER_HOST"
$remoteRepo = $env:ZO_CLUSTER_REPO_DIR
$sshPassword = [Environment]::GetEnvironmentVariable("ZO_CLUSTER_PASSWORD", "Process")
$hostKey = [Environment]::GetEnvironmentVariable("ZO_CLUSTER_HOSTKEY", "Process")

function Invoke-Remote {
    param([string]$Command)

    if ($sshPassword) {
        $args = @("-batch", "-ssh")
        if ($hostKey) { $args += @("-hostkey", $hostKey) }
        $args += @("-pw", $sshPassword, $target, $Command)
        & plink @args
    } else {
        & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 $target $Command
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Copy-Remote {
    param([string]$Local, [string]$Remote)

    if ($sshPassword) {
        $args = @("-batch")
        if ($hostKey) { $args += @("-hostkey", $hostKey) }
        $args += @("-pw", $sshPassword, $Local, "${target}:$Remote")
        & pscp @args
    } else {
        & scp -o ServerAliveInterval=30 -o ServerAliveCountMax=4 $Local "${target}:$Remote"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Remote copy failed with exit code $LASTEXITCODE"
    }
}

Write-Host "==> Sync repo to ${target}:${remoteRepo}"
Invoke-Remote "mkdir -p '$remoteRepo'"
$tmpTar = Join-Path ([System.IO.Path]::GetTempPath()) "zero-one-philyr-$([System.Guid]::NewGuid()).tar"
$excludeFile = Join-Path ([System.IO.Path]::GetTempPath()) "zero-one-philyr-excludes-$([System.Guid]::NewGuid()).txt"
@(
    "./.git",
    "./.env",
    "./.venv",
    "./apps/frontend/node_modules",
    "./experiments",
    "./hf_cache",
    "./slurm_logs",
    "./wandb"
) | Set-Content $excludeFile
tar --exclude-from=$excludeFile -cf $tmpTar -C $Root .
if ($LASTEXITCODE -ne 0) {
    throw "Local tar failed with exit code $LASTEXITCODE"
}
Remove-Item $excludeFile
Copy-Remote $tmpTar "$remoteRepo/repo.tar"
Remove-Item $tmpTar
Invoke-Remote "cd '$remoteRepo' && tar -xf repo.tar && rm repo.tar"
Invoke-Remote "cd '$remoteRepo' && find scripts -name '*.sh' -exec sed -i 's/\r$//' {} +"

Write-Host "==> Write cluster .env with local secrets (ignored by git)"
$tmpEnv = [System.IO.Path]::GetTempFileName()
Get-Content $EnvPath |
    Where-Object {
        $_ -match '^(ZO_|HF_TOKEN=|HF_HOME=|WANDB_)' -and
        $_ -notmatch '^ZO_CLUSTER_PASSWORD=' -and
        $_ -notmatch '^ZO_CLUSTER_HOSTKEY='
    } |
    Set-Content -Encoding Ascii $tmpEnv
$envText = [System.IO.File]::ReadAllText($tmpEnv) -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($tmpEnv, $envText, [System.Text.Encoding]::ASCII)
Copy-Remote $tmpEnv "$remoteRepo/.env"
Remove-Item $tmpEnv

Write-Host "==> Pre-stage GPU environment and base model on the login node"
Invoke-Remote "bash '$remoteRepo/scripts/leonardo_remote_prestage.sh'"

Write-Host "==> Submit short Leonardo smoke finetune"
$submitOut = uv run zo-cluster submit --config $Config 2>&1 | Out-String
Write-Host $submitOut
if ($LASTEXITCODE -ne 0) { throw "zo-cluster submit failed" }

$runId = if ($submitOut -match "run (\d{8}_\d{6}_\S+)") { $Matches[1] } else { $null }
$jobId = if ($submitOut -match "submitted SLURM job (\d+)") { $Matches[1] } else { $null }
if (-not $runId) { throw "Could not parse run id from submit output" }

if ($WaitAndUpload) {
    if (-not $env:ZO_CLUSTER_PROXY) {
        Write-Warning 'ZO_CLUSTER_PROXY is empty - GPU live wandb needs proxy (deck p.95).'
    }
    & (Join-Path $PSScriptRoot "leonardo_wait_upload.ps1") -RunId $runId -JobId $jobId
}
