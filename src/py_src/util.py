import os
import shutil
import re
import time
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, ClassifierMixin


flare_class_map = {'No Flare': 0, 'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
reverse_flare_class_map = {v: k for k, v in flare_class_map.items()}

goes_magnitude_map = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}


class ThresholdXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        probas = self.model.predict_proba(x)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


def create_dirs(path: str, range_:range) -> None:
    for i in range_:
        year_dir = os.path.join(path, str(i))
        os.makedirs(year_dir, exist_ok=True)

    return None


def move_file(origin_dir: str, destiny_dir: str, file_name: str | re.Pattern[str]) -> None:
    files_in_dir = os.listdir(origin_dir)
    found_file_name = None
    for file in files_in_dir:
        if isinstance(file_name, re.Pattern):
            if file_name.match(file):
                found_file_name = file
                break
        else:
            if file==file_name:
                found_file_name = file
                break
    try:
        full_origin_path = os.path.join(origin_dir, found_file_name)
        full_destiny_path = os.path.join(destiny_dir, found_file_name)
        shutil.move(full_origin_path, full_destiny_path)
    except TypeError as e:
        print(f'FILE ({file_name}) NOT FOUND')

    return None


def delete_file(dir_: str, file_name: str) -> None:
    full_path = os.path.join(dir_, file_name)
    os.remove(full_path)

    return None


def wait_download(file_path: str, file_name: str, timeout_seconds: int = 300) -> None:
    part_file_path = file_path + '.part'
    start_time = time.time()
    download_finished = False

    print(f"        -> Aguardando conclusão do download...")
    while time.time() - start_time < timeout_seconds:
        if not os.path.exists(part_file_path):
            time.sleep(1)
            if os.path.exists(file_path):
                print(f"        -> Download de '{file_name}' concluído com sucesso!")
                download_finished = True
                break

        time.sleep(1)

    if not download_finished:
        print(f"        -> ERRO: Timeout! O download de '{file_name}' demorou mais de {timeout_seconds}s.")

    return None


def create_df_model_input(slided_df_path: str, target_column: str, wanted_cols_start_with: str, resample_value: str,
                          resample_method: str) -> pd.DataFrame:

    slided_df = pd.read_parquet(slided_df_path)

    wanted_cols = [c for c in slided_df.columns if c.startswith(wanted_cols_start_with) or c == target_column]
    df_model_input = slided_df[wanted_cols]
    df_model_input = df_model_input.resample(resample_value).agg(resample_method).ffill().dropna()
    df_model_input[target_column] = df_model_input[target_column].astype(int)

    return df_model_input


def create_df_model_input_opt(slided_df_path, target_columns, wanted_cols_start_with):
    try:
        parquet_schema = pq.read_schema(slided_df_path)
        all_columns = parquet_schema.names
    except Exception as e:
        print(f"Erro ao ler o esquema do Parquet: {e}")
        all_columns = pd.read_parquet(slided_df_path, columns=[]).columns

    wanted_cols = [c for c in all_columns if c.startswith(wanted_cols_start_with) or c in target_columns]

    print(f"Carregando {len(wanted_cols)} colunas do arquivo Parquet...")

    try:
        df_model_input = pd.read_parquet(slided_df_path, columns=wanted_cols)
    except MemoryError:
        print("Erro de Memória: Mesmo carregando colunas selecionadas, a RAM estourou.")
        return None
    except Exception as e:
        print(f"Erro ao carregar colunas do Parquet: {e}")
        return None

    return df_model_input


def prepare_data(df_model_input: pd.DataFrame, target_class_col: str, lambda_function: callable,
                 train_pct: float, val_pct: float, test_pct: float = 0, target_flux_col: str = None) -> dict:
    dict_ = {
        'y': {},
        'x': {},
    }

    n = len(df_model_input)
    train_end = int(train_pct * n)
    val_end = int(n * (val_pct + train_pct))

    if target_flux_col is not None:
        dict_['flux'] = {}
        dict_['flux']['all'] = df_model_input[target_flux_col]

        dict_['flux']['train'] = dict_['flux']['all'].iloc[:train_end]
        dict_['flux']['val'] = dict_['flux']['all'].iloc[train_end:val_end]
        dict_['flux']['test'] = dict_['flux']['all'].iloc[val_end:]

        df_features = df_model_input.drop(columns=[target_class_col, target_flux_col], errors='ignore')
    else:
        df_features = df_model_input.drop(columns=[target_class_col], errors='ignore')

    dict_['y']['all'] = df_model_input[target_class_col].apply(lambda_function)
    dict_['x']['all'] = df_features

    dict_['x']['train'] = dict_['x']['all'].iloc[:train_end]
    dict_['y']['train'] = dict_['y']['all'].iloc[:train_end]

    dict_['x']['val'] = dict_['x']['all'].iloc[train_end:val_end]
    dict_['y']['val'] = dict_['y']['all'].iloc[train_end:val_end]

    dict_['x']['test'] = dict_['x']['all'].iloc[val_end:]
    dict_['y']['test'] = dict_['y']['all'].iloc[val_end:]

    return dict_


def parse_flare_class_expanded(class_expanded: str) -> float | None:
    if pd.isna(class_expanded) or not isinstance(class_expanded, str) or len(class_expanded) < 2:
        return None

    letter = class_expanded[0].upper()
    try:
        coef = float(class_expanded[1:])

        magnitude = goes_magnitude_map.get(letter)
        if magnitude:
            return coef * magnitude
        else:
            return None
    except ValueError:
        return None
