import numpy as np
from sklearn.linear_model import Lasso

def compressed_sensing_recovery(signal, measurement_ratio=0.5, alpha=1e-4, transform_type='dct'):
    """
    Compressed sensing recovery using L1 minimisation (Lasso).
    Returns: original signal, recovered signal, recovery error (MSE).
    """
    n = len(signal)
    if n < 5 or np.any(np.isnan(signal)):
        return signal, signal, 0.0  # fallback
    # Number of measurements
    m = max(int(np.ceil(measurement_ratio * n)), 1)
    # Random measurement matrix (Gaussian)
    np.random.seed(42)  # deterministic
    A = np.random.randn(m, n) / np.sqrt(m)
    # Measurements
    y = A @ signal
    # L1 recovery via Lasso
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
    lasso.fit(A, y)
    recovered = lasso.coef_
    mse = np.mean((signal - recovered) ** 2)
    return signal, recovered, mse

def compressed_sensing_anomaly_score(returns, measurement_ratio=0.5, alpha=1e-4, transform_type='dct'):
    """Return anomaly score = recovery MSE."""
    signal = returns.values.flatten()
    # Remove NaN and inf
    signal = signal[np.isfinite(signal)]
    if len(signal) < 5:
        return 0.0
    _, _, mse = compressed_sensing_recovery(signal, measurement_ratio, alpha, transform_type)
    return mse
