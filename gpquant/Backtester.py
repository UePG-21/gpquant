import numba as nb
import numpy as np
import pandas as pd


def _signal_to_asset(
    df: pd.DataFrame, signal: np.ndarray, init_cash: float, charge_ratio: float
) -> pd.Series:
    """
    @param df: market information including 'dt', 'C', 'A' and 'B'
    @param signal: trading decision at the end of the datetime (>0: long, <0: short, =0: hold)
    @param init_cash: initial cash
    @param charge_ratio: transaction cost = amount * charge ratio
    @return: asset: asset series with DatetimeIndex
    """
    if len(signal) != len(df):
        raise ValueError("signal must be the same length as df")
    sr_signal = pd.Series(signal)
    sr_long = sr_signal[sr_signal > 0]
    sr_short = sr_signal[sr_signal < 0]
    impact_cost = (sr_long * (df["A"] - df["C"])).fillna(0) + (
        sr_short * (df["B"] - df["C"])
    ).fillna(0)
    transaction_cost = (
        (sr_long * df["A"]).fillna(0) - (sr_short * df["B"]).fillna(0)
    ) * charge_ratio
    raw_position = sr_signal.cumsum() - sr_signal
    change = (df["C"] - df["C"].shift()).fillna(0)
    raw_return = raw_position * change
    sr_signal.index = df["dt"]
    return pd.Series(
        np.array(init_cash + (raw_return - impact_cost - transaction_cost).cumsum()),
        index=df["dt"],
    )


class Backtester:
    def __init__(self, factor_to_signal, signal_to_asset=_signal_to_asset) -> None:
        """
        Vectorized factor backtesting (factor -> signal -> asset)
        [factor] outcome of SyntaxTree.execute(X)
        [signal] trading decision at the end of the datetime (>0: long, <0: short, =0: hold)
        [asset] backtesting result of an account applying the strategy
        """
        self.f2s = factor_to_signal  # function
        self.s2a = signal_to_asset  # function

    def __call__(
        self, df_market, factor, init_cash, charge_ratio, **kwargs
    ) -> pd.Series:
        """
        @param df_market: market information including 'datetime', 'C', 'A' and 'B'
        @param factor: time series of factor with the same length as df_market
        @param init_cash: initial cash
        @param charge_ratio: transaction cost = amount * charge ratio
        @param kwargs: arguments except factor in factor_to_signal()
        @return: asset: time series of asset with DatetimeIndex
        """
        return self.s2a(df_market, self.f2s(factor, **kwargs), init_cash, charge_ratio)


@nb.jit(nopython=True)
def __limit_max_position(signal: np.ndarray, limit: int = 1) -> np.ndarray:
    # Process the signal so that each position is not greater than 0
    """auxiliary function, such that absolute value of each element in signal.cumsum() is not greater than limit"""
    sum_flag = 0
    for i, num in enumerate(signal):
        if abs(sum_flag + num) > limit:
            signal[i] = 0
            continue
        sum_flag += num
    return signal


# strategy (factor_to_signal)
def _strategy_quantile(
    factor: np.ndarray,
    d: int,
    o_upper: float,
    o_lower: float,
    c_upper: float,
    c_lower: float,
) -> np.ndarray:
    sr_factor = pd.Series(factor)
    sr_factor.fillna(method="ffill", inplace=True)
    sr_o_upper = sr_factor.rolling(d).quantile(o_upper)
    sr_o_lower = sr_factor.rolling(d).quantile(o_lower)
    sr_c_upper = sr_factor.rolling(d).quantile(c_upper)
    sr_c_lower = sr_factor.rolling(d).quantile(c_lower)
    signal = np.zeros((len(factor),))
    signal[sr_factor > sr_o_upper] = 1
    signal[sr_factor < sr_c_upper] = -1
    signal[sr_factor > sr_o_lower] = -1
    signal[sr_factor < sr_c_lower] = 1
    return __limit_max_position(signal)


# backtester
bt_quantile = Backtester(factor_to_signal=_strategy_quantile)


backtester_map = {"quantile": bt_quantile}
