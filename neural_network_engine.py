"""
Quant Research Neural Network Engine
Models 1–4 (A: NumPy Scratch, B: Advanced ML-style)
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# =========================
# CONFIG
# =========================
TICKER = "AAPL"
WINDOW = 5
TRAIN_SPLIT = 0.8
EPOCHS = 200
LR = 0.01

# =========================
# DATA
# =========================
def load_returns():
    df = yf.download(TICKER, period="3y")
    returns = df["Close"].pct_change().dropna().values
    return returns

def rolling_dataset(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y).reshape(-1, 1)

# =========================
# NUMPY NN
# =========================
class NumpyNN:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def relu(self, z): return np.maximum(0, z)
    def relu_d(self, z): return (z > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        return self.a1 @ self.W2 + self.b2

    def backward(self, X, y, y_hat):
        m = X.shape[0]
        dz2 = y_hat - y
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_d(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0, keepdims=True)

        self.W2 -= LR * dW2
        self.b2 -= LR * db2
        self.W1 -= LR * dW1
        self.b1 -= LR * db1

def train_numpy(model, X, y):
    losses = []
    for _ in range(EPOCHS):
        y_hat = model.forward(X)
        losses.append(np.mean((y_hat - y) ** 2))
        model.backward(X, y, y_hat)
    return losses

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    returns = load_returns()
    X, y = rolling_dataset(returns, WINDOW)

    # Flatten X for ML models
    X_flat = X.reshape(len(X), -1)

    split = int(TRAIN_SPLIT * len(X))
    X_train, y_train = X_flat[:split], y[:split]
    X_test, y_test = X_flat[split:], y[split:]

    # =====================
    # MODEL 1 — RETURNS
    # =====================
    nn = NumpyNN(WINDOW, 10)
    loss_1A = train_numpy(nn, X_train, y_train)

    rf_ret = RandomForestRegressor(n_estimators=200)
    rf_ret.fit(X_train, y_train.ravel())
    ret_pred = rf_ret.predict(X_test)

    # =====================
    # MODEL 2 — VOLATILITY (FIXED)
    # =====================
    realized_vol = np.array([
        np.std(returns[i:i+WINDOW])
        for i in range(len(returns) - WINDOW)
    ]).reshape(-1, 1)

    vol_train = realized_vol[:split]
    vol_test = realized_vol[split:]

    rf_vol = RandomForestRegressor(n_estimators=200)
    rf_vol.fit(X_train, vol_train.ravel())
    vol_pred = rf_vol.predict(X_test)

    # =====================
    # MODEL 3 — REGIME
    # =====================
    regime_labels = (vol_train.ravel() > np.median(vol_train)).astype(int)
    clf = LogisticRegression()
    clf.fit(X_train, regime_labels)
    regime = clf.predict(X_test)[-1]

    # =====================
    # MODEL 4 — SIGNAL
    # =====================
    signal_A = np.sign(ret_pred[-1]) / (vol_pred[-1] + 1e-6)
    signal_B = (0.3 if regime else 0.6) * signal_A

    print("\n=== FINAL OUTPUT ===")
    print("Return prediction:", ret_pred[-1])
    print("Volatility prediction:", vol_pred[-1])
    print("Regime:", "HIGH_VOL" if regime else "LOW_VOL")
    print("Signal A:", signal_A)
    print("Signal B:", signal_B)

    plt.plot(loss_1A)
    plt.title("Model 1A Training Loss")
    plt.show()