source activate "python3"
pip install --index-url https://build.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt --use-deprecated=legacy-resolver

jupyter nbconvert --execute --to notebook --inplace "0-setup.ipynb" --ExecutePreprocessor.kernel_name="python3"
jupyter nbconvert --execute --to notebook --inplace "1-eda-dev-data.ipynb" --ExecutePreprocessor.kernel_name="python3"
jupyter nbconvert --execute --to notebook --inplace "2-monitoring-results.ipynb" --ExecutePreprocessor.kernel_name="python3"
jupyter nbconvert --execute --to notebook --inplace "3-psi.ipynb" --ExecutePreprocessor.kernel_name="python3"
jupyter nbconvert --execute --to notebook --inplace "4-refit-original-models.ipynb" --ExecutePreprocessor.kernel_name="python3"
jupyter nbconvert --execute --to notebook --inplace "5-documentation-dev.ipynb" --ExecutePreprocessor.kernel_name="python3"

conda deactivate
