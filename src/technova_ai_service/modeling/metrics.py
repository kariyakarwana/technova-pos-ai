import numpy as np
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
