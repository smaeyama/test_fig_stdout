from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# プロットの基本設定です。
plt.rcParams.update({
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.family': 'sans-serif',
})

def _load_label_value(path: Path) -> pd.DataFrame:
    '''
    左列がラベル文字列、右列が数値の 2 列のテキストを読み込む関数です。

    Parameters
    ----------
    path : Path
        読み込むファイルのパスを渡します。

    Returns
    -------
    pandas.DataFrame
        ['label', 'value'] 列を持つ DataFrame を返します。
    '''
    try:
        df = pd.read_csv(
            path,
            sep=r'\s+',
            header=None,
            names=['label', 'equal', 'value', 'value2']
        )
        df['value'] = pd.to_numeric(df['value'], errors='raise')
        return df
    except Exception as e:
        raise ValueError(f'読み込みに失敗しました: {path}') from e

def plot_elt(data_dir: Path, pdf_out: Path) -> None:
    '''
    elt_coarse / medium / fine を描画して PDF 形式で保存します。

    Parameters
    ----------
    data_dir : Path
        *.dat ファイルが格納されたディレクトリ パスを渡します。

    pdf_out : Path
        出力 PDF ファイルのパスを渡します。
    '''
    filenames: List[str] = ['elt_coarse', 'elt_medium', 'elt_fine']
    titles: List[str] = [
        'Coasely-classified elapsed time',
        'Moderately-classified elapsed time',
        'Finely-classified elapsed time'
    ]
    dfs: List[pd.DataFrame] = [
        _load_label_value(data_dir / f'{t}.dat') for t in filenames
    ]
    max_n: int = max(len(df) for df in dfs)

    # レイアウト定数です。
    PAGE_W_IN, PAGE_H_IN = 8.27, 11.69
    LEFT, RIGHT, TOP = 0.8, 0.8, 1.0
    HSPACE = 1.2
    AX_H = 1.6

    # Figure を作成します。
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))

    # 左右余白を 0–1 の比率へ変換しておきます。
    left_ratio: float  = LEFT / PAGE_W_IN
    right_ratio: float = 1 - RIGHT / PAGE_W_IN
    full_width_ratio: float = right_ratio - left_ratio

    # 初期 y 位置（下端）を mm で算出
    current_y: float = PAGE_H_IN - TOP - AX_H

    for df, title in zip(dfs, titles, strict=True):

        n: int = len(df)

        # 棒幅を揃えるための補正処理です。
        width_ratio = full_width_ratio * (n / max_n)

        # (left, bottom, width, height) の Figure 座標を設定します。
        pos = (
            left_ratio,
            current_y / PAGE_H_IN,
            width_ratio,
            AX_H / PAGE_H_IN
        )
        ax = fig.add_axes(pos)

        # 棒グラフを描画します。
        x: NDArray[np.int_] = np.arange(n)
        ax.bar(
            x, df['value'],
            width=0.4,
            facecolor='none', edgecolor='#1f77b4', linewidth=0.5
        )
        ax.set_xticks(x, df['label'], rotation=-45, ha='left', va='top', fontsize=8)

        # 軸を設定します、
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Elapsed time [sec]', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(which='both', direction='in', labelsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)

        # 次のグラフの位置を更新します、
        current_y -= AX_H + HSPACE

    # ───── 保存 & クリーンアップ ───────────────────
    fig.savefig(str(pdf_out), format='pdf')
    plt.close(fig)

