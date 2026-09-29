import numpy as np
import pandas as pd
from numpy.typing import NDArray


def regression_metrics(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> dict[str, float]:
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denominator = max(float(np.sum(np.abs(actual))), 1e-9)
    wape = float(np.sum(np.abs(errors)) / denominator)
    return {"mae": mae, "rmse": rmse, "wape": wape}


def calculate_forecasting_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_train_seasonal: pd.Series | np.ndarray | None = None,
    seasonality: int = 7,
) -> dict[str, float | None]:
    """Calculate standard evaluation metrics for sales forecasting.

    Primary:
    - WAPE (Weighted Absolute Percentage Error): sum(|y - y_hat|) / sum(y)
    - MASE (Mean Absolute Scaled Error): MAE / MAE(seasonal_naive_train)

    Secondary:
    - MAE: mean(|y - y_hat|)
    - RMSE: sqrt(mean((y - y_hat)^2))

    Args:
        y_true: Actual ground-truth sales.
        y_pred: Predicted sales.
        y_train_seasonal: Optional training series used to compute seasonal naive denominator for MASE.
        seasonality: Seasonal period for MASE scaling (default 7 days).

    Returns:
        Dictionary of computed metric values.
    """
    s_true = pd.Series(y_true).astype(float)
    s_pred = pd.Series(y_pred).astype(float)

    valid_mask = s_true.notna() & s_pred.notna()
    yt = s_true[valid_mask].to_numpy()
    yp = s_pred[valid_mask].to_numpy()

    if len(yt) == 0:
        return {"mae": np.nan, "rmse": np.nan, "wape": np.nan, "mase": np.nan}

    abs_errors = np.abs(yt - yp)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # WAPE: sum(|y - y_hat|) / sum(y)
    total_actual = float(np.sum(yt))
    wape = float(np.sum(abs_errors) / total_actual) if total_actual > 0 else np.nan

    # MASE: MAE / mean(|y_train[t] - y_train[t - seasonality]|)
    mase = np.nan
    if y_train_seasonal is not None:
        s_train = pd.Series(y_train_seasonal).dropna().astype(float).to_numpy()
        if len(s_train) > seasonality:
            scale = np.mean(np.abs(s_train[seasonality:] - s_train[:-seasonality]))
            if scale > 0:
                mase = float(mae / scale)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 6),
        "mase": round(mase, 6) if not np.isnan(mase) else None,
    }

