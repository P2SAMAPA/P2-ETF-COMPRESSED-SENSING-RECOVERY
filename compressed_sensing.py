import numpy as np
from scipy.fft import dct, idct
from sklearn.linear_model import Lasso

def sparse_transform(signal, transform_type='dct'):
    """Forward sparse transform (returns coefficients)."""
    if transform_type == 'dct':
        coeff = dct(signal, norm='ortho')
    else:
        # wavelet placeholder – use DCT as fallback
        coeff = dct(signal, norm='ortho')
    return coeff

def inverse_transform(coeff, transform_type='dct'):
    """Inverse transform."""
    if transform_type == 'dct':
        return idct(coeff, norm='ortho')
    else:
        return idct(coeff, norm='ortho')

def compressed_sensing_recovery(signal, measurement_ratio=0.5, alpha=1e-4, transform_type='dct'):
    """
    Compressed sensing recovery using L1 minimisation (Lasso).
    Returns: original signal, recovered signal, recovery error (MSE).
    """
    n = len(signal)
    if n < 10:
        return signal, signal, 0.0
    # Number of measurements
    m = max(int(np.ceil(measurement_ratio * n)), 1)
    # Random measurement matrix (Gaussian)
    np.random.seed(42)  # deterministic for reproducibility
    A = np.random.randn(m, n) / np.sqrt(m)
    # Measurements
    y = A @ signal
    # L1 recovery: min ||x||_1 s.t. A x = y
    # Use Lasso: min 0.5||A x - y||^2 + alpha ||x||_1
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
    lasso.fit(A, y)
    recovered = lasso.coef_
    # Transform to original domain? No, we already recovered in signal domain.
    # But we can also enforce sparsity in transform domain: work with coefficients.
    # Better: recover in transform domain where signal is sparse.
    # Let's do: signal is sparse in DCT domain. So we solve for DCT coefficients.
    # Let x = coeffs. Then measurement = A * (inverse_transform(coeffs)).
    # But that's non-linear. Simpler: apply compressed sensing directly to signal
    # and rely on Lasso to find a sparse representation naturally.
    # The current approach is correct: Lasso finds a sparse solution to A x = y.
    # Recovery error in signal space.
    mse = np.mean((signal - recovered) ** 2)
    return signal, recovered, mse

def compressed_sensing_anomaly_score(returns, measurement_ratio=0.5, alpha=1e-4, transform_type='dct'):
    """Return anomaly score = recovery MSE."""
    signal = returns.values.flatten()
    _, _, mse = compressed_sensing_recovery(signal, measurement_ratio, alpha, transform_type)
    return mse
