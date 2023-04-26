import pandas as pd
from gpquant.SymbolicRegressor import SymbolicRegressor

if __name__ == "__main__":
    file_path = "data.csv"
    df = pd.read_csv(file_path, parse_dates=["dt"])
    slippage = 0.001
    df["A"] = df["C"] * (1 + slippage)
    df["B"] = df["C"] * (1 - slippage)
    print(df)

    sr = SymbolicRegressor(
        population_size=2000,
        tournament_size=20,
        generations=50,
        stopping_criteria=2,
        p_crossover=0.6,
        p_subtree_mutate=0.2,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(6, 8),
        init_method="half and half",
        function_set=[],
        variable_set=["O", "H", "L", "C", "V"],
        const_range=(1, 20),
        ts_const_range=(1, 20),
        build_preference=[0.75, 0.75],
        metric="sharpe ratio",
        transformer="quantile",
        transformer_kwargs={
            "init_cash": 5000,
            "charge_ratio": 0.00002,
            "d": 15,
            "o_upper": 0.8,
            "c_upper": 0.6,
            "o_lower": 0.2,
            "c_lower": 0.4,
        },
        parsimony_coefficient=0.005,
    )

    sr.fit(df.iloc[:400], df["C"].iloc[:400])
    print(sr.score(df.iloc[400:800], df["C"].iloc[400:800]))
