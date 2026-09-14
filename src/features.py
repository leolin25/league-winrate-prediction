import pandas as pd

"""
This function takes in the df of data and adds the engineered features as new columns into a new df
"""
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_engineered = df.copy()
    df_engineered['gold_diff'] = df_engineered['blueGoldDiff']
    df_engineered['xp_diff'] = df_engineered['blueExperienceDiff']
    df_engineered['kill_diff'] = df_engineered['blueKills'] - df_engineered['redKills']
    df_engineered['cs_diff'] = df_engineered['blueCSPerMin'] - df_engineered['redCSPerMin']
    df_engineered['objective_diff'] = df_engineered['blueEliteMonsters'] - df_engineered['redEliteMonsters']
    df_engineered['first_blood'] = df_engineered['blueFirstBlood'] # 1 if blue fb, 0 o/w
    df_engineered['jungle_diff'] = df_engineered['blueTotalJungleMinionsKilled'] - df_engineered['redTotalJungleMinionsKilled']
    df_engineered['tower_diff'] = df_engineered['blueTowersDestroyed'] - df_engineered['redTowersDestroyed']
    df_engineered['herald_diff'] = df_engineered['blueHeralds'] # 1 if blue has herald, 0 o/w
    df_engineered['vision_diff'] = df_engineered['blueWardsPlaced'] + df_engineered['blueWardsDestroyed'] - df_engineered['redWardsPlaced'] - df_engineered['redWardsDestroyed']

    return df_engineered

"""
Choose the selected features and target for the model training, returns X, y
"""
def select_model_features(df: pd.DataFrame, selected_features: list, target: str='blueWins') -> tuple:
    # ensure the target column exists
    if target not in df.columns:
        raise ValueError(f'Target column {target} was not found')

    X = df[selected_features]
    y = df[target]

    return X, y