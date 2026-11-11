# MLP Lab 6 Project

## Files
- `MLP_Lab6_Core.ipynb` — complete core notebook with preprocessing, MLP design/training, parameter calculations, evaluation, Plotly visualizations, architecture experiments, and self-learning Q&A.
- `app.py` — interactive Streamlit dashboard with Plotly visualizations, adjustable MLP settings, metrics, architecture, parameter analysis, training curves, confusion matrix, ROC curve, dataset explorer, and self-learning section.
- `requirements.txt` — Python dependencies.

## Run locally
Python 3.10–3.11 recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Notebook
Open `MLP_Lab6_Core.ipynb` in Jupyter or Google Colab and run cells from top to bottom. TensorFlow is required. The notebook downloads no data: it uses scikit-learn's bundled Wisconsin Diagnostic Breast Cancer dataset.

## Submission
Rename files to match the lab's required convention using your own name, last three register-number digits, and program number. Update the notebook cover metadata if required by your institution.

## Important
This is an educational classification exercise, not a clinically validated diagnostic model.
