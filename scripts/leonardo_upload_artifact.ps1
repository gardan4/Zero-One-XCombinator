param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$HubModelId = "XCombinator/leonardo-smoke-qwen-0.5b-lora"
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
$cmd = "bash '$remoteRepo/scripts/leonardo_upload_artifact.sh' '$RunId' '$HubModelId'"

if ($sshPassword) {
    $args = @("-batch", "-ssh")
    if ($hostKey) { $args += @("-hostkey", $hostKey) }
    $args += @("-pw", $sshPassword, $target, $cmd)
    & plink @args
} else {
    & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 $target $cmd
}
if ($LASTEXITCODE -ne 0) {
    throw "Remote upload failed with exit code $LASTEXITCODE"
}
