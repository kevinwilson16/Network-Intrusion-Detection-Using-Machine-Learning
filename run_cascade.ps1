.\venv\Scripts\Activate.ps1

Write-Host "--- DATA PREPROCESSING START ---"
python src\data\preprocess_cicids2017.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\data\preprocess_unsw.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\data\cross_dataset_utils.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "--- CIC-IDS2017 TRAINING START ---"
python src\models\train_binary.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\models\train_multiclass.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\models\train_unsupervised.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\evaluation\evaluate_hybrid.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "--- UNSW-NB15 TRAINING START ---"
python src\models\train_unsw_binary.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\models\train_unsw_multiclass.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\models\train_unsw_unsupervised.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\evaluation\evaluate_unsw_hybrid.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "--- CROSS DATASET EVALUATION START ---"
python src\evaluation\evaluate_cross_dataset.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "--- ARTIFACT GENERATION START ---"
python src\evaluation\generate_artifacts.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "--- CASCADE COMPLETE SUCCESSFULLY! ---"
