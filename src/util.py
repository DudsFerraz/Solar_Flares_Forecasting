import os
import shutil
import re
import time
import pandas as pd



flare_class_map = {'No Flare': 0, 'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
reverse_flare_class_map = {v: k for k, v in flare_class_map.items()}



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

def prepare_data(df_model_input: pd.DataFrame, target_column: str, lambda_function: callable, train_pct: float,
                 val_pct: float , test_pct: float=0) -> dict:

    dict_ = {
            'y':{},
             'x':{}
    }

    dict_['x']['all'] = df_model_input.drop(columns=[target_column])
    dict_['y']['all'] = df_model_input[target_column].apply(lambda_function)

    n = len(dict_['x']['all'])
    train_end = int(train_pct * n)
    val_end = int(n * (val_pct + train_pct))

    dict_['x']['train'] = dict_['x']['all'].iloc[:train_end]
    dict_['y']['train'] = dict_['y']['all'].iloc[:train_end]

    dict_['x']['val'] = dict_['x']['all'].iloc[train_end:val_end]
    dict_['y']['val'] = dict_['y']['all'].iloc[train_end:val_end]

    dict_['x']['test'] = dict_['x']['all'].iloc[val_end:]
    dict_['y']['test'] = dict_['y']['all'].iloc[val_end:]

    return dict_

