import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def add_metrics_to_df(df_rm:pd.DataFrame, df_md:pd.DataFrame, df_pwldr:pd.DataFrame):
    # Add RP, WS and EVPI to pwldr dataframe
    df_pivot = df_rm.pivot(
        index='idx_p', 
        columns='metric', 
        values='value'
    )
    df_pwldr = df_pwldr[df_pwldr["displace_func"] == "local_search"]
    df_pwldr = df_pwldr[df_pwldr["metric"] == "dr_pwldr"]
    df_pwldr = df_pwldr[df_pwldr["nb"] == 1]

    df_pwldr["RP"] = df_pwldr['idx_p'].map(df_pivot['reopt_std'])
    df_pwldr["WS"] = df_pwldr['idx_p'].map(df_pivot['ws'])
    df_pwldr["LDR"] = df_pwldr['idx_p'].map(df_pivot['dr_ldr'])
    df_pwldr["EVPI"] = df_pwldr["RP"] - df_pwldr["WS"]
    df_pwldr["EVPI_nb_0"] = df_pwldr["LDR"] - df_pwldr["WS"]
    df_pwldr["EVPI_DR"] = df_pwldr["value_opt"] - df_pwldr["WS"]
    df_pwldr["Gain"] = (df_pwldr["EVPI_nb_0"] - df_pwldr["EVPI_DR"]) / df_pwldr["EVPI"]

    df_1 = pd.merge(df_pwldr, df_md, on=['idx_p', 'idx_v'], how='left')

    df_final = df_1[["idx_p","idx_v","nb","Gain","variable",
                 "v1","v2","v3","v4","v5","v6","v7","v8",
                 "v9","v10","v11","v12","v13","v14","v15"]]

    return df_final

def merge_df(list_df):
    max_idx = max(list_df[0]["idx_p"])

    df_final = list_df[0]
    for df in list_df[1:]:
        max_idx_df = max(df["idx_p"])
        df["idx_p"] += max_idx
        max_idx += max_idx_df

    df_final = pd.concat(list_df)

    return df_final

def open_data():
    sp_regular_metrics = "../../../data/shipment_planning_regular_metrics.csv"
    sp_metadata = "../../../data/shipment_planning_pwldr_metadata.csv"
    sp_pwldr = "../../../data/shipment_planning_pwldr_metrics.csv"

    sp_df_rm = pd.read_csv(sp_regular_metrics)
    sp_df_md = pd.read_csv(sp_metadata)
    sp_df_pwldr = pd.read_csv(sp_pwldr)

    ce_regular_metrics = "../../../data/capacity_expansion_regular_metrics.csv"
    ce_metadata = "../../../data/capacity_expansion_pwldr_metadata.csv"
    ce_pwldr = "../../../data/capacity_expansion_pwldr_metrics.csv"

    ce_df_rm = pd.read_csv(ce_regular_metrics)
    ce_df_md = pd.read_csv(ce_metadata)
    ce_df_pwldr = pd.read_csv(ce_pwldr)

    nf_regular_metrics = "../../../data/capacity_expansion_regular_metrics.csv"
    nf_metadata = "../../../data/capacity_expansion_pwldr_metadata.csv"
    nf_pwldr = "../../../data/capacity_expansion_pwldr_metrics.csv"

    nf_df_rm = pd.read_csv(nf_regular_metrics)
    nf_df_md = pd.read_csv(nf_metadata)
    nf_df_pwldr = pd.read_csv(nf_pwldr)

    df_sp = add_metrics_to_df(sp_df_rm, sp_df_md, sp_df_pwldr)
    df_ce = add_metrics_to_df(ce_df_rm, ce_df_md, ce_df_pwldr)
    df_nf = add_metrics_to_df(nf_df_rm, nf_df_md, nf_df_pwldr)
    return merge_df([df_sp, df_ce, df_nf])

def fit_regression(df: pd.DataFrame):
    X = df[[f'v{i}' for i in range(1, 16)]].copy()
    y = df['Gain'].to_numpy()

    # Remove colunas sem variância
    variances = X.var()
    X = X[variances[variances > 0].index]

    # Normalização
    X_mean = X.mean() # Média do treino
    X_std = X.std()   # Desvio padrão do treino

    # Normalização
    X = (X - X_mean) / X_std

    X = X.to_numpy()

    # Adiciona intercepto
    X = np.column_stack([np.ones(len(X)), X])

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    feature_names = ['intercept'] + list(variances[variances > 0].index)
    coef = pd.Series(coef, index=feature_names)

    return coef, X_mean, X_std

def train_model(test_size=0.2, random_state=42, save_path="model_params.json"):
    df = open_data()
    X = df[[f'v{i}' for i in range(1, 16)]].copy()
    y = df['Gain'].to_numpy()

    variances = X.var()
    X = X[variances[variances > 0].index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_mean = X_train.mean()
    X_std = X_train.std()

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    feature_names = ['intercept'] + list(variances[variances > 0].index)
    coef_series = pd.Series(coef, index=feature_names)

    y_train_pred = X_train @ coef
    y_test_pred = X_test @ coef

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    mae_train = np.mean(np.abs(y_train - y_train_pred))
    mae_test = np.mean(np.abs(y_test - y_test_pred))

    metrics = {
        "train": {
            "r2": r2_train,
            "rmse": rmse_train,
            "mae": mae_train
        },
        "test": {
            "r2": r2_test,
            "rmse": rmse_test,
            "mae": mae_test
        }
    }

    model_params = {
        "coefficients": coef_series.to_dict(),
        "mean": X_mean.to_dict(),
        "std": X_std.to_dict(),
        "features": list(X_mean.index),
        "metrics": metrics
    }

    with open(save_path, "w") as f:
        json.dump([model_params], f, indent=4)

    print("Modelo treinado com sucesso!")
    print("Métricas de Teste:")
    for k, v in metrics["test"].items():
        print(f"{k}: {v:.6f}")

    return coef_series, metrics

train_model()