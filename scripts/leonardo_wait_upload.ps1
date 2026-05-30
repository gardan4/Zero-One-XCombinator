param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$JobId = "",
    [string]$HubModelId = "XCombinator/leonardo-smoke-qwen-0.5b-lora",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvPath = Join-Path $Root ".env"

Get-Content $EnvPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
    $key, $value = $line.Split("=", 2)
    [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), "Process")
}

$target = "$env:ZO_CLUSTER_USER@$env:ZO_CLUSTER_HOST"
$hostKey = [Environment]::GetEnvironmentVariable("ZO_CLUSTER_HOSTKEY", "Process")
$sshPassword = [Environment]::GetEnvironmentVariable("ZO_CLUSTER_PASSWORD", "Process")
$remoteRepo = $env:ZO_CLUSTER_REPO_DIR
$experiments = $env:ZO_CLUSTER_EXPERIMENTS_DIR

function Invoke-Remote {
    param([string]$Command)
    $args = @("-batch", "-ssh")
    if ($hostKey) { $args += @("-hostkey", $hostKey) }
    $args += @("-pw", $sshPassword, $target, $Command)
    & plink @args
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $Command" }
    return $LASTEXITCODE
}

function Invoke-RemoteCapture {
    param([string]$Command)
    $args = @("-batch", "-ssh")
    if ($hostKey) { $args += @("-hostkey", $hostKey) }
    $args += @("-pw", $sshPassword, $target, $Command)
    $out = & plink @args 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $Command`n$out" }
    return ($out | Out-String).Trim()
}

Write-Host "==> Waiting for SLURM job $(if ($JobId) { $JobId } else { '(any of yours)' })"
while ($true) {
    $q = Invoke-RemoteCapture "squeue --me --noheader 2>/dev/null || true"
    if ($JobId) {
        if ($q -notmatch "\b$JobId\b") { break }
    } elseif (-not $q.Trim()) {
        break
    }
    Write-Host "  still running... ($(Get-Date -Format 'HH:mm:ss'))"
    Start-Sleep -Seconds $PollSeconds
}

$logGlob = "$remoteRepo/slurm_logs/${RunId}-*.out"
Write-Host "==> Tail SLURM log ($logGlob)"
$tail = Invoke-RemoteCapture "ls -1t $logGlob 2>/dev/null | head -1 | xargs -r tail -n 80"
Write-Host $tail

if ($tail -notmatch "cuda_available=True") {
    Write-Warning "GPU check string not found in log — confirm nvidia-smi / torch block in slurm output."
}
if ($tail -notmatch "'loss'|loss") {
    Write-Warning "No training loss in log tail — job may have failed before train()."
}

Write-Host "==> Upload artifacts to Hugging Face (login node)"
& (Join-Path $PSScriptRoot "leonardo_upload_artifact.ps1") -RunId $RunId -HubModelId $HubModelId
Write-Host "Done. Run $RunId  artifacts: $experiments/$RunId/artifacts"
