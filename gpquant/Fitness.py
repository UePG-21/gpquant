import numpy as np
import pandas as pd


class Fitness:
    def __init__(self, function, greater_is_better: bool) -> None:
        self.function = function
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args) -> float:
        return self.function(*args)


def _ann_return(useless_var, asset: pd.Series) -> float:
    return (asset.values[-1] / asset.values[0]) ** (250 / len(asset)) - 1


def _sharpe_ratio(close: pd.Series, asset: pd.Series, r_f: float | None = 0.0) -> float:
    # factors with no trading are considered the worst factors -> sharpe = np.nan
    close_copy = close.copy()
    close_copy.index = asset.index
    benchmark_return = _ann_return(None, close_copy) if r_f is None else r_f
    volatility = np.std(asset / asset.shift() - 1) * np.sqrt(250)
    excess_return = _ann_return(None, asset) - benchmark_return
    if excess_return > 0:
        return excess_return / volatility if volatility else np.nan
    else:
        # adjust sharpe: if excess return < 0, then sharpe = excess return * volatility
        return excess_return * volatility if volatility else np.nan


def _mean_absolute_error(y: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(np.abs(y_pred - y))


def _mean_square_error(y: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(((y_pred - y) ** 2))


def _direction_accuracy(close: pd.Series, factor) -> float:
    sr_factor = pd.Series(factor)
    close_direction = np.where(close - close.shift() > 0, 1, 0)
    factor_direction = np.where(sr_factor - sr_factor.shift() > 0, 1, 0)
    return np.sum((close_direction == factor_direction)) / len(factor)


# fitness indicator
ann_return = Fitness(_ann_return, greater_is_better=True)
sharpe_ratio = Fitness(_sharpe_ratio, greater_is_better=True)
mean_absolute_error = Fitness(_mean_absolute_error, greater_is_better=False)
mean_square_error = Fitness(_mean_square_error, greater_is_better=False)
direction_accuracy = Fitness(_direction_accuracy, greater_is_better=True)


fitness_map = {
    "annual return": ann_return,
    "sharpe ratio": sharpe_ratio,
    "mean absolute error": mean_absolute_error,
    "mean square error": mean_square_error,
    "direction accuracy": direction_accuracy,
}
