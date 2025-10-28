import os
import shutil
import re
import time

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


flare_class_map = {'No Flare': 0, 'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
reverse_flare_class_map = {v: k for k, v in flare_class_map.items()}