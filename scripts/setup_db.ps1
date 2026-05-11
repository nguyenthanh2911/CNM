Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Import-DotEnv {
  param(
    [Parameter(Mandatory=$true)][string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_.Trim()
    if ($line.Length -eq 0) { return }
    if ($line.StartsWith('#')) { return }

    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { return }

    $key = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim()

    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }

    if ($key.Length -gt 0) {
      [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
    }
  }
}

function Get-EnvOrDefault {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Default
  )

  $value = [System.Environment]::GetEnvironmentVariable($Name, 'Process')
  if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
  return $value
}

Import-DotEnv -Path (Join-Path $PSScriptRoot '..\.env')

$postgresUser = Get-EnvOrDefault -Name 'POSTGRES_USER' -Default 'sepsis_user'
$postgresDb = Get-EnvOrDefault -Name 'POSTGRES_DB' -Default 'sepsis_db'
$serviceName = Get-EnvOrDefault -Name 'POSTGRES_SERVICE' -Default 'postgres'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw 'docker is not installed or not available in PATH.'
}

Write-Host "Ensuring '$serviceName' container is running..."
$containerId = (docker compose ps -q $serviceName) 2>$null
if ([string]::IsNullOrWhiteSpace($containerId)) {
  docker compose up -d $serviceName | Out-Host
}

$schemaPath = Join-Path $PSScriptRoot '..\docs\database_schema.sql'
if (-not (Test-Path -LiteralPath $schemaPath)) {
  throw "Schema file not found: $schemaPath"
}

Write-Host 'Waiting for Postgres to be ready...'
$maxAttempts = 30
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
  $ready = $true
  try {
    docker compose exec -T $serviceName pg_isready -U $postgresUser -d $postgresDb | Out-Null
  } catch {
    $ready = $false
  }

  if ($ready) {
    break
  }

  if ($attempt -eq $maxAttempts) {
    throw "Postgres not ready after $maxAttempts attempts."
  }

  Start-Sleep -Seconds 1
}

Write-Host 'Applying schema...'
Get-Content -LiteralPath $schemaPath -Raw |
  docker compose exec -T $serviceName psql -U $postgresUser -d $postgresDb -v ON_ERROR_STOP=1 | Out-Host

Write-Host 'Database setup complete!'
