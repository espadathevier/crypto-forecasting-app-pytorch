# Crypto Return Forecaster

This project turns the notebook workflow into a small Streamlit app you can demo live.

## What it does

- Pulls daily crypto market data from Alpha Vantage
- Engineers technical indicators used in the original notebook
- Trains a GRU + attention model on returns
- Shows next-day return prediction, test metrics, and visualizations

## Project Files

- `app.py` - Streamlit app
- `requirements.txt` - Python dependencies
- `.gitignore` - Git hygiene for local artifacts and secrets

## Run locally

1. Create a virtual environment:

```powershell
py -3 -m venv .venv
```

2. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the app:

```powershell
streamlit run app.py
```

5. In the app sidebar, enter your Alpha Vantage API key and click `Fetch data and train model`.

## Demo Tips

- Use `BTC` and `USD` first because that matches your notebook.
- Keep epochs around `20-40` for a quicker live demo.
- Talk through the predicted next-day return, the estimated next close, and the actual vs predicted chart.

## Upload to GitHub

If this repository is not connected to GitHub yet, run:

```powershell
git init
git add .
git commit -m "Add Streamlit crypto forecasting app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

If the repo already exists on GitHub and you only need to push your changes:

```powershell
git add .
git commit -m "Add Streamlit crypto forecasting app"
git push
```

## Notes

- Alpha Vantage free keys are rate-limited, so repeated refreshes may temporarily fail.
- The app retrains on demand so you can show interaction with the model instead of only static output.
