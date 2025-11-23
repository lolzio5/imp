# Run quick smoke tests for all registered models in this repo
# Usage examples:
#  - Quick smoke (fast):  ./scripts/run_all_models.ps1 -Mode quick
#  - Full evaluation (expensive): ./scripts/run_all_models.ps1 -Mode full

param(
    [ValidateSet('quick','full')]
    [string]$Mode = 'quick',
    # name of the conda environment to activate before running
    [string]$CondaEnv = 'imp-py38'
)

# Try to activate the requested conda environment. If activation isn't available
# in this shell (e.g. `conda init powershell` wasn't run), fall back to using
# `conda run -n <env> --no-capture-output python` when invoking Python.
$useFallback = $false
Write-Output "Activating conda environment '$CondaEnv'..."
try {
    conda activate $CondaEnv
} catch {
    Write-Warning "Could not run 'conda activate'. Make sure you ran 'conda init powershell' or run this script from an Anaconda/Miniconda prompt. Using fallback."
    $useFallback = $true
}

if (-not $useFallback) {
    if ($env:CONDA_DEFAULT_ENV -ne $CondaEnv) {
        Write-Warning "Conda environment activation didn't set CONDA_DEFAULT_ENV. Using fallback invocation."
        $useFallback = $true
    }
}

if ($useFallback) {
    $pythonCmdPrefix = "conda run -n $CondaEnv --no-capture-output python"
} else {
    $pythonCmdPrefix = "python"
}

# list of registered models discovered in the codebase
$models = @(
    'protonet',
    'kmeans-refine',
    'dp-means-hard',
    'kmeans-distractor',
    'imp',
    'crp',
    'map-dp',
    'soft-nn'
)

# Common args you can tweak
$data = 'omniglot'
$nshot = 1
$nclassesTrain = 20
$nclassesEval = 5
$resultsRoot = './results'

if ($Mode -eq 'quick') {
    # quick smoke: few episodes, no long training (use small numbers)
    $commonArgs = "--dataset $data --nshot $nshot --nclasses-train $nclassesTrain --nclasses-eval $nclassesEval --results $resultsRoot --num-eval-episode 10"
} else {
    # full runs: adjust as desired (this will be slow)
    $commonArgs = "--dataset $data --nshot $nshot --nclasses-train $nclassesTrain --nclasses-eval $nclassesEval --results $resultsRoot --num-eval-episode 100"
}

# Ensure results directory exists
if (-not (Test-Path -Path $resultsRoot)) { New-Item -ItemType Directory -Path $resultsRoot | Out-Null }

foreach ($m in $models) {
    $modelResults = Join-Path $resultsRoot $m
    if (-not (Test-Path -Path $modelResults)) { New-Item -ItemType Directory -Path $modelResults | Out-Null }

    $args = "--model $m $commonArgs"
    $log = Join-Path $modelResults "run.log"

    Write-Output "Running model: $m  (mode=$Mode) -> logging to $log"

    # run the command, redirect stdout+stderr to log; runs sequentially
    # Use either activated python or the 'conda run' fallback prefix configured above.
    $cmd = "$pythonCmdPrefix run_eval.py $args"
    Write-Output "Executing: $cmd"
    Invoke-Expression "$cmd 2>&1 | Tee-Object -FilePath $log"

    # small pause between runs
    Start-Sleep -Seconds 2
}

Write-Output "All done. Check $resultsRoot for per-model subfolders and logs."