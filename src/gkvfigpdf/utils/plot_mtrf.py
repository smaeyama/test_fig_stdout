from pathlib import Path
from typing import Final, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# レイアウト用のグリッドの設定です。
N_COLS, N_ROWS = 2, 6

# A4 縦向きのページ設定です。
PAGE_W_IN, PAGE_H_IN = 8.27, 11.69
# マージンを設定します。
LEFT, RIGHT = 1.0, 0.6
TOP, BOTTOM = 0.8, 0.9
HSPACE, VSPACE = 0.24, 0.36

# 描画対象列を設定します。
COLS: Final[List[int]] = list(range(3, 14))

# mtr の Y 軸ラベルです。
YLABELS_MTR: Final[List[str]] = [
    r'$B\;[B_{\mathrm{ref}}]$',
    r'$\mathrm{d}B/\mathrm{d}x \;[B_{\mathrm{ref}}/L_{\mathrm{ref}}]$',
    r'$\mathrm{d}B/\mathrm{d}y \;[B_{\mathrm{ref}}/L_{\mathrm{ref}}]$',
    r'$\mathrm{d}B/\mathrm{d}z \;[B_{\mathrm{ref}}]$',
    r'$g^{xx}$',
    r'$g^{xy}$',
    r'$g^{xz}\;[L_{\mathrm{ref}}^{-1}]$',
    r'$g^{yy}$',
    r'$g^{yz}\;[L_{\mathrm{ref}}^{-1}]$',
    r'$g^{zz}\;[L_{\mathrm{ref}}^{-2}]$',
    r'Jacobian $[L_{\mathrm{ref}}]$',
]

# mtf の Y 軸ラベルです。
YLABELS_MTF: Final[List[str]] = [
    r'$B\;[B_{\mathrm{ref}}]$',
    r'$\mathrm{d}B/\mathrm{d}\rho \;[B_{\mathrm{ref}}/L_{\mathrm{ref}}]$',
    r'$\mathrm{d}B/\mathrm{d}\theta \;[B_{\mathrm{ref}}/L_{\mathrm{ref}}]$',
    r'$\mathrm{d}B/\mathrm{d}\zeta \;[B_{\mathrm{ref}}]$',
    r'$g^{\rho\rho}$',
    r'$g^{\rho\theta}$',
    r'$g^{\rho\zeta}\;[L_{\mathrm{ref}}^{-1}]$',
    r'$g^{\theta\theta}$',
    r'$g^{\theta\zeta}\;[L_{\mathrm{ref}}^{-1}]$',
    r'$g^{\zeta\zeta}\;[L_{\mathrm{ref}}^{-2}]$',
    r'Jacobian$_{\rho\theta\zeta}\;[L_{\mathrm{ref}}]$',
]

# テーマを設定します。
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

def _plot(dat_path: Path, pdf_out: Path, xlabel: str, ylabels: List[str]) -> None:
    '''
    mtr と mtf の描画両方に用いられる関数です。
    dat_path で指定されたデータ ファイルを読み込み、
    2 列 6 行の図を PDF として生成し、pdf_out へ保存します。

    Parameters
    ----------
    dat_path : Path
        1 列目が x 軸、続く列がプロット対象となる数値データ ファイルのパスを渡します。

    pdf_out : Path
        出力される PDF ファイルのパスを渡します。

    xlabel : str
        X 軸ラベルを渡します。

    ylabels : list[str]
        各サブプロットの Y 軸ラベルを格納したリストです。
    '''
    data = np.loadtxt(dat_path)
    z = data[:, 0]

    with PdfPages(pdf_out) as pdf:
        fig, axes = plt.subplots(
            N_ROWS, N_COLS,
            figsize=(PAGE_W_IN, PAGE_H_IN),
            sharex=True
        )

        fig.subplots_adjust(
            left=LEFT/PAGE_W_IN, right=1-RIGHT/PAGE_W_IN,
            top=1-TOP/PAGE_H_IN, bottom=BOTTOM/PAGE_H_IN,
            wspace=HSPACE, hspace=VSPACE
        )

        for idx, (col, yl) in enumerate(zip(COLS, ylabels, strict=True)):
            ax = axes.flat[idx]
            ax.plot(z, data[:, col - 1], lw=0.5, marker='+', markersize=4)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks(np.arange(-3, 4, 1))
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.tick_params(labelsize=7, direction='in')
            ax.tick_params(axis='x', labelbottom=True)

            ax.yaxis.get_offset_text().set_size(7)
            ax.yaxis.get_offset_text().set_fontstyle('italic')

        axes.flat[-1].axis('off')
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

def plot_mtr(dat_path: Path, pdf_out: Path) -> None:
    '''
    mtr.dat を読み込み、グラフを PDF として保存します。

    Parameters
    ----------
    dat_path : Path
        mtr.dat へのパスを渡します。

    pdf_out : Path
        出力される PDF ファイルのパスを渡します。
    '''
    _plot(dat_path, pdf_out, r'Field-aligned coordinate $z$', YLABELS_MTR)

def plot_mtf(dat_path: Path, pdf_out: Path) -> None:
    '''
    mtf.dat を読み込み、グラフを PDF として保存します。

    Parameters
    ----------
    dat_path : Path
        mtf.dat へのパスを渡します。

    pdf_out : Path
        出力される PDF ファイルのパスを渡します。
    '''
    _plot(dat_path, pdf_out, r'Poloidal angle $\theta$', YLABELS_MTF)
