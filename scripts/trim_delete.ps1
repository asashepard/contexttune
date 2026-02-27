$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
  throw "Python was not found in PATH."
}

Write-Host "Running hard delete trim..."
if ($python.Name -eq 'py.exe' -or $python.Name -eq 'py') {
  & $python.Source -3 scripts/hard_trim_repo.py
} else {
  & $python.Source scripts/hard_trim_repo.py
}

Write-Host "Done."
