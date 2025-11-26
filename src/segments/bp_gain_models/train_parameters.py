import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def add_metrics_to_df(df_rm, df_md, df_pwldr):
    # Add nb = 0 to pwldr dataframe
    df1_unique = df_pwldr[['idx_p', 'idx_v', 'displace_func']].drop_duplicates()
    df_rm['metric'] = df_rm['metric'].str.replace('_ldr', '_pwldr')
    df_rm_filter = df_rm[df_rm["metric"].isin(["obj_pwldr", "reopt_pwldr", "dr_pwldr"])]
    new_lines = pd.merge(df1_unique, df_rm_filter, on='idx_p')

    new_lines['nb'] = 0
    new_lines['time_uni'] = 0
    new_lines['time_opt'] = 0
    new_lines['value_uni'] = new_lines['value']
    new_lines['value_opt'] = new_lines['value']

    new_lines = new_lines.drop(columns=['value'])
    new_lines = new_lines[df_pwldr.columns]

    df_pwldr = pd.concat([df_pwldr, new_lines], ignore_index=True)

    # Add RP, WS and EVPI to pwldr dataframe
    df_pivot = df_rm.pivot(
        index='idx_p', 
        columns='metric', 
        values='value'
    )

    df_pwldr["RP"] = df_pwldr['idx_p'].map(df_pivot['reopt_std'])
    df_pwldr["WS"] = df_pwldr['idx_p'].map(df_pivot['ws'])
    df_pwldr["EVPI"] = df_pwldr["RP"] - df_pwldr["WS"]

    df_1 = pd.merge(df_pwldr, df_md, on=['idx_p', 'idx_v'], how='left')
    df_1["gain"] = - (df_1["value_opt"] - df_1["value_uni"]) / df_1["EVPI"]

    df_1 = df_1[df_1["displace_func"] == "local_search"]
    df_1 = df_1[df_1["metric"] == "obj_pwldr"]

    df_baseline = df_1[df_1['nb'] == 0][[
        'idx_p', 'idx_v', 'value_opt'
    ]]

    df_baseline = df_baseline.rename(columns={
        'value_opt': 'value_opt_nb0',
    })
    df_1 = pd.merge(
        df_1,
        df_baseline,
        on=['idx_p', 'idx_v'],
        how='left'
    )
    df_1['gain'] = (df_1['value_opt_nb0'] - df_1['value_opt']) / (df_1['EVPI'])
    df_final = df_1[["idx_p","idx_v","nb","gain","variable",
                 "v1","v2","v3","v4","v5","v6","v7","v8",
                 "v9","v10","v11","v12","v13","v14","v15"]]

    return df_final

def merge_df(list_df):
    max_idx = max(list_df[1]["idx_p"])

    df_final = list_df[1]
    for df in list_df[1:]:
        max_idx_df = max(df["idx_p"])
        df["idx_p"] += max_idx
        max_idx += max_idx_df

    df_final = pd.concat(list_df)

    return df_final

def open_data():
    sp_regular_metrics = "data/shipment_planning_regular_metrics.csv"
    sp_metadata = "data/shipment_planning_pwldr_metadata.csv"
    sp_pwldr = "data/shipment_planning_pwldr_metrics.csv"

    sp_df_rm = pd.read_csv(sp_regular_metrics)
    sp_df_md = pd.read_csv(sp_metadata)
    sp_df_pwldr = pd.read_csv(sp_pwldr)

    df_md_fixed = sp_df_md[["idx_p","idx_v","variable", "v5", "v10", "v15"]]
    df_md_fixed['variable'] = df_md_fixed['variable'].str.replace('Truncated{', '')
    df_md_fixed['variable'] = df_md_fixed['variable'].str.split('{').str[0]

    df_md_fixed['v_sum'] = df_md_fixed['v5'] + df_md_fixed['v10'] + df_md_fixed['v15']
    df_md_fixed['max_v_sum'] = df_md_fixed.groupby('idx_p')['v_sum'].transform('max')

    is_normal = df_md_fixed['variable'] == 'Normal'
    is_max_sum = df_md_fixed['v_sum'] == df_md_fixed['max_v_sum']

    df_md_fixed.loc[is_normal & is_max_sum, 'variable'] = 'Normal(50, 40)'
    df_md_fixed.loc[is_normal & ~is_max_sum, 'variable'] = 'Normal(50, 15)'

    sp_df_md["variable"] = df_md_fixed['variable']

    ce_regular_metrics = "data/capacity_expansion_regular_metrics.csv"
    ce_metadata = "data/capacity_expansion_pwldr_metadata.csv"
    ce_pwldr = "data/capacity_expansion_pwldr_metrics.csv"

    ce_df_rm = pd.read_csv(ce_regular_metrics)
    ce_df_md = pd.read_csv(ce_metadata)
    ce_df_pwldr = pd.read_csv(ce_pwldr)

    df_md_fixed = ce_df_md[["idx_p","idx_v","variable", "v5", "v10", "v15"]]
    df_md_fixed['variable'] = df_md_fixed['variable'].str.replace('Truncated{', '')
    df_md_fixed['variable'] = df_md_fixed['variable'].str.split('{').str[0]

    df_md_fixed['v_sum'] = df_md_fixed['v5'] + df_md_fixed['v10'] + df_md_fixed['v15']
    df_md_fixed['v_sum_rank'] = df_md_fixed.groupby('idx_p')['v_sum'].rank(ascending=False, method='first')

    is_normal = df_md_fixed['variable'] == 'Normal'
    is_top_2_sum = df_md_fixed['v_sum_rank'] <= 2 

    df_md_fixed.loc[is_normal & is_top_2_sum, 'variable'] = 'Normal(50, 40)'
    df_md_fixed.loc[is_normal & ~is_top_2_sum, 'variable'] = 'Normal(50, 15)'

    ce_df_md["variable"] = df_md_fixed['variable']

    df_ce = add_metrics_to_df(ce_df_rm, ce_df_md, ce_df_pwldr)
    df_sp = add_metrics_to_df(sp_df_rm, sp_df_md, sp_df_pwldr)
    df_final = merge_df([df_ce, df_sp])
    return df_final

def fit_regression(df:pd.DataFrame):
    X = df[[f'v{i}' for i in range(1, 16)]]
    y = df['gain']

    variances = X.var()
    valid_columns = variances[variances > 0].index
    X = X[valid_columns]
    X = X.to_numpy()
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    calculated_coefs = pd.Series(coef, index=valid_columns)
    all_v = [f'v{i}' for i in range(1, 16)]
    full_coefs = calculated_coefs.reindex(all_v, fill_value=0)

    return full_coefs

def train_model():
    df = open_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Total de dados: {len(df)}")
    print(f"Dados de Treino: {len(train_df)}")
    print(f"Dados de Teste: {len(test_df)}")
    print("-" * 30)

    coef_series = fit_regression(train_df)

    # Create columns
    X_test = test_df[[f'v{i}' for i in range(1, 16)]].copy()
    y_test = test_df['gain']
        
    X_test = X_test[coef_series.index]
    predictions = X_test @ coef_series
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("Resultados da Avaliação no Conjunto de Teste:")
    print(f"R² (R-squared): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

    coef_series.to_json('src/segments/bp_gain_models/params.json', orient='values')

train_model()