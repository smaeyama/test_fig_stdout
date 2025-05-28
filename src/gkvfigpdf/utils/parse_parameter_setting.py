import re
from pathlib import Path

def parse_parameters(log_path: Path) -> tuple[int, int, str]:
    '''
    GKV のログ ファイルから以下の 3 つのパラメータを抽出します：
        - nprocs    (プロセス数)
        - global_ny (全体の Y 方向の格子点数)
        - calc_type (計算の種類)

    Parameters
    ----------
    log_path : Path
        ログ ファイルのパスを渡します。

    Returns
    -------
    tuple[int, int, str]
        抽出された nprocs, global_ny, calc_type のタプルを返します。
    '''
    nprocs = None
    global_ny = None
    calc_type = None

    with log_path.open(encoding='utf-8') as f:
        for line in f:
            if 'nprocs' in line and 'rank' in line:
                match = re.search(r'#?\s*nprocs\s*,\s*rank\s*=\s*(\d+)', line)
                if match:
                    nprocs = int(match.group(1))

            elif 'global_ny' in line:
                match = re.search(r'#?\s*global_ny\s*=\s*(\d+)', line)
                if match:
                    global_ny = int(match.group(1))

            elif 'Type of calc' in line:
                match = re.search(r'#?\s*Type of calc\.\s*[:=]\s*(\w+)', line)
                if match:
                    calc_type = match.group(1)

    if nprocs is None or global_ny is None or calc_type is None:
        print(f'nprocs = {nprocs}, global_ny = {global_ny}, calc_type = {calc_type}')
        raise ValueError('ログ ファイルからのパラメーター抽出に失敗しました。')

    return nprocs, global_ny, calc_type
