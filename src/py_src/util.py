import os
import shutil
import re
import time
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np

flare_class_map = {'No Flare': 0, 'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
reverse_flare_class_map = {v: k for k, v in flare_class_map.items()}

goes_magnitude_map = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}


def create_dirs(path: str, range_: range) -> None:
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
            if file == file_name:
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


def prepare_data_global_secure(df_model_input: pd.DataFrame, target_class_col: str,
                               time_col: str, lambda_function: callable,
                               train_pct: float, val_pct: float,
                               target_flux_col: str = None, purge_hours: int = 24) -> dict:
    # Ordena rigorosamente pelo tempo
    df = df_model_input.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    train_end = int(train_pct * n)
    val_end = int(n * (val_pct + train_pct))

    # 1. Faz o Split Bruto
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # 2. Aplica o Purge Gap (Remove amostras da Val/Test que sobrepõem o Treino/Val)
    purge_td = pd.Timedelta(hours=purge_hours)

    val_cutoff = train_df[time_col].max() + purge_td
    val_df = val_df[val_df[time_col] > val_cutoff]

    if not test_df.empty:
        test_cutoff = val_df[time_col].max() + purge_td
        test_df = test_df[test_df[time_col] > test_cutoff]

    # 3. Monta o Dicionário de Retorno
    dict_ = {'y': {}, 'x': {}}

    columns_to_drop = [target_class_col, time_col]
    if target_flux_col:
        columns_to_drop.append(target_flux_col)
        dict_['flux'] = {
            'train': train_df[target_flux_col],
            'val': val_df[target_flux_col],
            'test': test_df[target_flux_col] if not test_df.empty else None
        }

    # Popula X e Y aplicando a lambda
    for split_name, split_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        if split_df.empty:
            continue

        dict_['y'][split_name] = split_df[target_class_col].apply(lambda_function)
        dict_['x'][split_name] = split_df.drop(columns=columns_to_drop, errors='ignore')

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
