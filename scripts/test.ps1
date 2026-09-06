# Run the test suite in the pinned environment. This is THE way to run the tests.
#
#   .\scripts\test.ps1                    # whole suite
#   .\scripts\test.ps1 -k resume          # anything after the script name goes to pytest
#   .\scripts\test.ps1 tests\test_guards.py -x
#
# Creates .venv on the declared python series if it is absent or on the wrong
# series, installs the pinned extra when pyproject.toml has changed, and runs
# pytest inside it. Typing `pytest` directly instead uses whatever interpreter and
# whatever numpy happen to be on PATH -- on this laptop that is python 3.13 with
# numpy 2.2.6, against a pinned 3.11 and 1.26.4. tests/test_environment.py is the
# backstop that catches that; this script is what makes it not happen.
#
# The python series is read from docker/requirements.txt rather than hardcoded, so
# there is one source of truth for it.

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Die($message) {
    Write-Host "test.ps1: FATAL: $message" -ForegroundColor Red
    exit 1
}

function Invoke-Native {
    <#
    Run a native executable, failing on its EXIT CODE rather than on whatever it
    writes to stderr.

    Windows PowerShell 5.1 wraps every stderr line from a native command in an
    ErrorRecord, so under `$ErrorActionPreference = 'Stop'` an ordinary pip
    warning ("There was an error checking the latest version of pip") terminates
    the script. Exit codes are the only reliable signal here.
    #>
    param(
        [Parameter(Mandatory)][string]$What,
        [Parameter(Mandatory)][string]$Exe,
        [string[]]$Arguments = @()
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Exe @Arguments } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { Die "$What failed (exit $LASTEXITCODE)" }
}

# --------------------------------------------------------------------------
# what interpreter does this project want?
# --------------------------------------------------------------------------

$requirements = Get-Content "docker\requirements.txt" -Raw
$match = [regex]::Match($requirements, '(?m)^#\s*python:\s*(\d+)\.(\d+)\.\d+\s*$')
if (-not $match.Success) { Die "docker\requirements.txt has no '# python: X.Y.Z' line" }
$series = "$($match.Groups[1].Value).$($match.Groups[2].Value)"

function Get-Series($exe, $preArgs = @()) {
    # The probe deliberately contains NO quote characters. Windows PowerShell 5.1
    # does not escape embedded double quotes when invoking a native executable, so
    # a -c argument containing them reaches python as mangled source and every
    # interpreter looks absent. Printing two integers and joining them here avoids
    # the problem entirely.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $exe @preArgs -c 'import sys; print(sys.version_info[0], sys.version_info[1])'
        if ($LASTEXITCODE -eq 0 -and $out) {
            return (($out | Select-Object -First 1).Trim() -split '\s+') -join '.'
        }
    } catch {
    } finally {
        $ErrorActionPreference = $previous
    }
    return $null
}

function Find-BasePython {
    # The py launcher is the reliable way to pick a series on Windows.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        if ((Get-Series 'py' @("-$series")) -eq $series) {
            return @{ Exe = 'py'; PreArgs = @("-$series") }
        }
    }
    foreach ($candidate in @("python$series", 'python', 'python3')) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            if ((Get-Series $candidate) -eq $series) {
                return @{ Exe = $candidate; PreArgs = @() }
            }
        }
    }
    return $null
}

# --------------------------------------------------------------------------
# venv
# --------------------------------------------------------------------------

$venv = if ($env:CLEFT_VENV) { $env:CLEFT_VENV } else { Join-Path $repo '.venv' }
$vpy = Join-Path $venv 'Scripts\python.exe'

if ((Test-Path $vpy) -and ((Get-Series $vpy) -ne $series)) {
    Write-Host "test.ps1: $venv is python $(Get-Series $vpy), this project wants $series - recreating"
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $vpy)) {
    $base = Find-BasePython
    if (-not $base) {
        Die @"
no python $series available.

Tried: py -$series, python$series, python, python3.

  winget install Python.Python.$series
  or https://www.python.org/downloads/

Do NOT substitute another series. numpy 1.26.4 publishes no wheels beyond 3.12,
so installing the test extra cannot work on 3.13, and the cluster image ships $series.
"@
    }
    Write-Host "test.ps1: creating $venv from $($base.Exe) $($base.PreArgs -join ' ') (python $series)"
    Invoke-Native 'venv creation' $base.Exe (@($base.PreArgs) + @('-m', 'venv', $venv))
    Invoke-Native 'pip upgrade' $vpy @(
        '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet', '--disable-pip-version-check'
    )
}

# --------------------------------------------------------------------------
# dependencies, reinstalled only when the pins change
# --------------------------------------------------------------------------

$stamp = Join-Path $venv '.cleft-pins'
$want = (Get-FileHash 'pyproject.toml' -Algorithm SHA256).Hash
$have = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { '' }

if ($have -ne $want) {
    Write-Host 'test.ps1: installing pinned dependencies'
    Invoke-Native 'dependency install' $vpy @(
        '-m', 'pip', 'install', '-e', '.[test]', '--quiet', '--disable-pip-version-check'
    )
    # Stamped only after a successful install, so an interrupted one reinstalls
    # next time rather than being remembered as done.
    Set-Content -Path $stamp -Value $want -Encoding ascii
}

# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

Write-Host "test.ps1: $((Get-Series $vpy) -as [string]) -> $(& $vpy -VV)"

# pytest's own exit code is the result, so it is propagated rather than turned
# into a Die: 1 means tests failed, 5 means none were collected.
$previous = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try { & $vpy -m pytest @PytestArgs } finally { $ErrorActionPreference = $previous }
exit $LASTEXITCODE
