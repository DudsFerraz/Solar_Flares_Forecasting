import os
import shutil
import re
from dotenv import load_dotenv

def move_files(path_: str, begin_year_: int, end_year_: int) -> None:
    files_in_dir = os.listdir(path_)

    for y in range(begin_year_, end_year_+1):
        for m in range(1, 13):
            m_str = f"{m:02d}"
            pattern_str = f"^g\\d+_xrs_1m_3s_{y}{m_str}01_{y}{m_str}\\d{{2}}\\.csv$"
            pattern = re.compile(pattern_str)

            found_file_name = None
            for file_name in files_in_dir:
                if pattern.match(file_name):
                    found_file_name = file_name
                    break

            if found_file_name is None:
                print(f"ERRO: Arquivo não encontrado, pulando mes {m} de {y}")
                continue

            full_path = os.path.join(path_, found_file_name)
            destiny_path = os.path.join(path_, f"{y}")
            shutil.move(full_path, os.path.join(destiny_path, found_file_name))

    return None

def check_files(path_: str, begin_year_: int, end_year_: int) -> None:
    for y in range(begin_year_, end_year_+1):
        year_dir = os.path.join(path_, str(y))
        files_in_dir = os.listdir(year_dir)
        for m in range(1, 13):
            m_str = f"{m:02d}"

            if y <= 1985:
                pattern_str = f"^g\\d+_xrs_1m_3s_{y}{m_str}01_{y}{m_str}\\d{{2}}\\.csv$"
                pattern = re.compile(pattern_str)
            else:
                pattern_str = f"^g\\d+_xrs_1m_{y}{m_str}01_{y}{m_str}\\d{{2}}\\.csv$"
                pattern = re.compile(pattern_str)

            match = False
            for file in files_in_dir:
                if pattern.match(file):
                    match = True
                    break
            if not match:
                print(f"AVISO: Arquivo faltante! {m} de {y}")

    return None

def create_dirs(path_: str, begin_year_: int, end_year_: int) -> None:

    for y in range(begin_year_,end_year_+1):
        year_dir = os.path.join(path_, str(y))
        os.makedirs(year_dir, exist_ok=True)

    return None
