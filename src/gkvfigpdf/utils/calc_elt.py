'''
calc_elt.sh を Python に移植したスクリプトです。
ログファイル末尾 80 行から coarse / medium / fine の
経過時間データを抽出して保存します。
'''
from pathlib import Path
from typing import Final, Iterable, List, Sequence, Tuple

# 定数の定義です。
_COARSE:   Final[List[Tuple[int, int]]] = [(3, 14)]
_MEDIUM:   Final[List[Tuple[int, int]]] = [
    (6, 7), (18, 35), (72, 79), (14, 14)
]
_FINE:     Final[List[Tuple[int, int]]] = [
    (6, 7), (18, 20), (39, 47), (22, 24), (48, 59),
    (29, 29), (60, 62), (31, 31), (63, 65),
    (33, 34), (66, 68), (72, 79), (14, 14)
]

def _select_lines(
        lines: Sequence[str],
        ranges: Sequence[Tuple[int, int]]) -> Iterable[str]:
    '''
    指定範囲の行を抽出するジェネレータです。

    Parameters
    ----------
    lines : Sequence[str]
        ログファイル全行を格納したシーケンスを渡します。

    ranges : Sequence[Tuple[int, int]]
        抽出したい行範囲 (1 始まり, 両端含む) を渡します。

    Yields
    ------
    str
        抽出された行を返します。
    '''
    for i0, i1 in ranges:
        for idx in range(i0 - 1, i1):
            yield lines[idx]

def calc_elt(log_filepath: Path, out_dir: Path) -> None:
    '''
    calc_elt.sh 相当の処理を実行します。

    Parameters
    ----------
    log_file : Path
        解析対象の GKV ログ ファイルのパスを渡します。

    out_dir : Path
        elt_*.dat を保存するディレクトリを渡します。
        出力ファイルは elt_coarse.dat, elt_medium.dat, elt_fine.dat です。
    '''
    out_dir.mkdir(exist_ok=True, parents=True)

    tail_lines: List[str] = log_filepath.read_text().splitlines()[-80:]

    for name, rng in (
        ('elt_coarse', _COARSE), ('elt_medium', _MEDIUM), ('elt_fine', _FINE)):
        out_path: Path = out_dir / f'{name}.dat'
        selected = '\n'.join(_select_lines(tail_lines, rng))
        out_path.write_text(selected, encoding='utf-8')
