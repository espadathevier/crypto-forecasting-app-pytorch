import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


st.set_page_config(
    page_title="Crypto Return Forecaster",
    page_icon="📈",
    layout="wide",
)

torch.manual_seed(42)
np.random.seed(42)

FEATURE_COLUMNS = [
    "volume",
    "MA50",
    "MACD",
    "MACD_signal",
    "MACD_diff",
    "RSI14",
    "ATR14",
    "OBV",
    "close_roll_std_5",
    "close_roll_mean_20",
    "dayofweek_sin",
    "dayofweek_cos",
    "Bollinger_PercB",
]


@dataclass
class TrainingArtifacts:
    df: pd.DataFrame
    test_frame: pd.DataFrame
    test_predictions: np.ndarray
    test_truth: np.ndarray
    next_return: float
    latest_close: float
    metrics: dict


class ForecastGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.dropout(out)
        attn_out, _ = self.attention(out, out, out)
        last_step = attn_out[:, -1, :]
        return self.fc(last_step)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_alpha_vantage_data(symbol: str, market: str, api_key: str) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol.upper(),
        "market": market.upper(),
        "apikey": api_key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "Note" in payload:
        raise ValueError(payload["Note"])
    if "Error Message" in payload:
        raise ValueError(payload["Error Message"])
    if "Time Series (Digital Currency Daily)" not in payload:
        raise ValueError("Alpha Vantage did not return daily crypto data.")

    raw = payload["Time Series (Digital Currency Daily)"]
    df = pd.DataFrame(raw).T

    market_code = market.upper()
    column_map = {
        f"1a. open ({market_code})": "open",
        f"2a. high ({market_code})": "high",
        f"3a. low ({market_code})": "low",
        f"4a. close ({market_code})": "close",
        "5. volume": "volume",
    }
    df = df.rename(columns=column_map)
    missing = [column for column in column_map.values() if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from API response: {missing}")

    df = df[list(column_map.values())].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().astype(float)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["MA10"] = ta.trend.sma_indicator(features["close"], window=10)
    features["MA50"] = ta.trend.sma_indicator(features["close"], window=50)
    features["EMA10"] = ta.trend.ema_indicator(features["close"], window=10)

    macd = ta.trend.MACD(features["close"])
    features["MACD"] = macd.macd()
    features["MACD_signal"] = macd.macd_signal()
    features["MACD_diff"] = macd.macd_diff()

    features["RSI14"] = ta.momentum.rsi(features["close"], window=14)
    features["ATR14"] = ta.volatility.average_true_range(
        features["high"], features["low"], features["close"], window=14
    )
    features["OBV"] = ta.volume.on_balance_volume(features["close"], features["volume"])

    features["close_roll_mean_5"] = features["close"].rolling(5).mean()
    features["close_roll_std_5"] = features["close"].rolling(5).std()
    features["close_roll_mean_20"] = features["close"].rolling(20).mean()
    features["close_roll_std_20"] = features["close"].rolling(20).std()

    features["dayofweek"] = features.index.dayofweek
    features["dayofweek_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
    features["dayofweek_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)

    features["MA20"] = features["close"].rolling(window=20).mean()
    features["20STD"] = features["close"].rolling(window=20).std()
    features["Upper_Band"] = features["MA20"] + 2 * features["20STD"]
    features["Lower_Band"] = features["MA20"] - 2 * features["20STD"]
    features["Bollinger_PercB"] = (
        (features["close"] - features["Lower_Band"])
        / (features["Upper_Band"] - features["Lower_Band"])
    )

    features["return"] = features["close"].pct_change()
    features = features.dropna().copy()
    return features


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def build_loaders(
    feature_frame: pd.DataFrame,
    seq_len: int,
    batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    X = feature_frame[FEATURE_COLUMNS].values
    y = feature_frame["return"].values

    n_samples = len(feature_frame)
    train_end = max(int(n_samples * 0.64), seq_len + 1)
    val_end = max(int(n_samples * 0.80), train_end + seq_len + 1)
    val_end = min(val_end, n_samples - 1)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    if min(len(X_train), len(X_val), len(X_test)) <= seq_len:
        raise ValueError("Not enough data after splitting. Reduce sequence length or use a larger dataset.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_len)

    tensors = {
        "X_train": torch.tensor(X_train_seq, dtype=torch.float32),
        "y_train": torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1),
        "X_val": torch.tensor(X_val_seq, dtype=torch.float32),
        "y_val": torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(-1),
        "X_test": torch.tensor(X_test_seq, dtype=torch.float32),
        "y_test": torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(-1),
        "X_all_scaled": X_all_scaled,
    }

    loaders = {
        "train_loader": DataLoader(TensorDataset(tensors["X_train"], tensors["y_train"]), batch_size=batch_size, shuffle=False),
        "val_loader": DataLoader(TensorDataset(tensors["X_val"], tensors["y_val"]), batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(TensorDataset(tensors["X_test"], tensors["y_test"]), batch_size=batch_size, shuffle=False),
    }

    test_frame = feature_frame.iloc[val_end + seq_len :].copy()
    return {**tensors, **loaders}, test_frame


def train_and_evaluate(
    df: pd.DataFrame,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    patience: int,
) -> TrainingArtifacts:
    prepared, test_frame = build_loaders(df, seq_len=seq_len, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastGRU(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val_loss = math.inf
    patience_counter = 0

    progress_bar = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in prepared["train_loader"]:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(prepared["train_loader"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in prepared["val_loader"]:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(prepared["val_loader"].dataset)
        scheduler.step(val_loss)

        progress_bar.progress((epoch + 1) / epochs)
        status.text(
            f"Epoch {epoch + 1}/{epochs} | train loss {train_loss:.6f} | val loss {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()
    progress_bar.empty()
    status.empty()

    with torch.no_grad():
        test_preds = model(prepared["X_test"].to(device)).cpu().numpy().flatten()
        val_preds = model(prepared["X_val"].to(device)).cpu().numpy().flatten()
        last_sequence = prepared["X_all_scaled"][-seq_len:]
        last_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        next_return = float(model(last_tensor).cpu().numpy().flatten()[0])

    test_truth = prepared["y_test"].numpy().flatten()
    val_truth = prepared["y_val"].numpy().flatten()

    metrics = {
        "validation_rmse": float(np.sqrt(mean_squared_error(val_truth, val_preds))),
        "validation_mae": float(mean_absolute_error(val_truth, val_preds)),
        "validation_directional_accuracy": float(np.mean(np.sign(val_preds) == np.sign(val_truth))),
        "test_rmse": float(np.sqrt(mean_squared_error(test_truth, test_preds))),
        "test_mae": float(mean_absolute_error(test_truth, test_preds)),
        "test_directional_accuracy": float(np.mean(np.sign(test_preds) == np.sign(test_truth))),
    }

    return TrainingArtifacts(
        df=df,
        test_frame=test_frame,
        test_predictions=test_preds,
        test_truth=test_truth,
        next_return=next_return,
        latest_close=float(df["close"].iloc[-1]),
        metrics=metrics,
    )


def render_overview(data: pd.DataFrame) -> None:
    st.subheader("Market Data")
    st.line_chart(data[["close"]], height=280)

    latest = data.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Close", f"${latest['close']:,.2f}")
    c2.metric("Latest Volume", f"{latest['volume']:,.0f}")
    c3.metric("Rows After Features", f"{len(data)}")

    with st.expander("Preview engineered features"):
        st.dataframe(data[["close", "volume", "MA50", "RSI14", "MACD", "Bollinger_PercB"]].tail(12))


def render_results(artifacts: TrainingArtifacts) -> None:
    st.subheader("Model Outputs")

    predicted_price = artifacts.latest_close * (1 + artifacts.next_return)
    direction = "Up" if artifacts.next_return >= 0 else "Down"

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Next-Day Return", f"{artifacts.next_return:.2%}")
    c2.metric("Estimated Next Close", f"${predicted_price:,.2f}")
    c3.metric("Signal", direction)

    m1, m2, m3 = st.columns(3)
    m1.metric("Validation RMSE", f"{artifacts.metrics['validation_rmse']:.4f}")
    m2.metric("Test RMSE", f"{artifacts.metrics['test_rmse']:.4f}")
    m3.metric("Test Direction Accuracy", f"{artifacts.metrics['test_directional_accuracy']:.2%}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(artifacts.test_frame.index, artifacts.test_truth, label="Actual Return", linewidth=1.5)
    ax.plot(artifacts.test_frame.index, artifacts.test_predictions, label="Predicted Return", linewidth=1.5)
    ax.set_title("Test Set: Actual vs Predicted Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    scatter_fig, scatter_ax = plt.subplots(figsize=(5, 5))
    scatter_ax.scatter(artifacts.test_truth, artifacts.test_predictions, alpha=0.6)
    min_val = min(np.min(artifacts.test_truth), np.min(artifacts.test_predictions))
    max_val = max(np.max(artifacts.test_truth), np.max(artifacts.test_predictions))
    scatter_ax.plot([min_val, max_val], [min_val, max_val], "r--")
    scatter_ax.set_title("Prediction Scatter Plot")
    scatter_ax.set_xlabel("Actual Return")
    scatter_ax.set_ylabel("Predicted Return")
    scatter_ax.grid(True, alpha=0.3)
    st.pyplot(scatter_fig)

    download_frame = artifacts.test_frame[["close", "return"]].copy()
    download_frame["predicted_return"] = artifacts.test_predictions
    download_frame["predicted_close"] = download_frame["close"] * (1 + download_frame["predicted_return"])
    st.download_button(
        label="Download test predictions as CSV",
        data=download_frame.to_csv().encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


st.title("Crypto Return Forecaster")
st.caption("Interactive Streamlit wrapper for your Alpha Vantage + GRU attention notebook.")

with st.sidebar:
    st.header("Inputs")
    api_key = st.text_input("Alpha Vantage API Key", type="password", help="Free API keys work, but rate limits are strict.")
    symbol = st.text_input("Symbol", value="BTC")
    market = st.text_input("Market", value="USD")

    st.header("Training Controls")
    seq_len = st.slider("Sequence length", min_value=10, max_value=60, value=20, step=5)
    epochs = st.slider("Epochs", min_value=10, max_value=120, value=40, step=10)
    hidden_dim = st.select_slider("Hidden dimension", options=[32, 64, 96, 128], value=64)
    num_layers = st.select_slider("GRU layers", options=[1, 2, 3], value=2)
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.3, step=0.1)
    learning_rate = st.select_slider("Learning rate", options=[0.0005, 0.001, 0.002, 0.005], value=0.001)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
    patience = st.slider("Early stopping patience", min_value=5, max_value=25, value=12, step=1)

    train_clicked = st.button("Fetch data and train model", type="primary", use_container_width=True)

st.markdown(
    """
This demo keeps the core notebook idea, but makes it presentable:
- Pulls daily crypto data from Alpha Vantage.
- Engineers technical indicators used by the notebook.
- Trains a GRU + attention model on returns.
- Shows next-day prediction, test metrics, and charts you can talk through live.
"""
)

if not api_key:
    st.info("Enter an Alpha Vantage API key in the sidebar to run the app.")
elif train_clicked:
    try:
        with st.spinner("Fetching data and building features..."):
            raw_df = fetch_alpha_vantage_data(symbol=symbol, market=market, api_key=api_key)
            feature_df = engineer_features(raw_df)

        render_overview(feature_df)

        with st.spinner("Training model..."):
            artifacts = train_and_evaluate(
                df=feature_df,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                patience=patience,
            )

        render_results(artifacts)
    except Exception as exc:
        st.error(f"Unable to run the demo: {exc}")
