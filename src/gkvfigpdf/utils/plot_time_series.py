from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import LogFormatter
from matplotlib.backends.backend_pdf import PdfPages

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

# レイアウト設定です [inch]。
PAGE_W_IN, PAGE_H_IN = 8.27, 11.69
LEFT, RIGHT = 1.2, 2.2
TOP, BOTTOM = 0.8, 0.9
HSPACE      = 0.28

def semilogy_formatter(ax: Axes, *, offset_fontsize: int = 7) -> None:
    '''
    Y 軸を対数スケールにしてオフセット文字を調整する関数です。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        描画対象のサブプロットを渡します。

    offset_fontsize : int, optional
        オフセット テキストのフォント サイズを設定します。
    '''
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatter(base=10))
    # オフセット テキストのフォント サイズを調整します。
    ax.yaxis.get_offset_text().set_fontsize(offset_fontsize)

def plot_time_series(global_ny: int, show_men: bool,
                    data_dir: Path, pdf_out: Path) -> None:
    '''
    時系列のグラフを最大 3 段で描画し PDF 出力する関数です。

    Parameters
    ----------
    global_ny : int
        プロットする m_y の最大番号を渡します。

    show_men : bool
        True の場合 3 段目のグラフを描画します。

    data_dir : Path
        *.dat ファイル群が置かれたディレクトリを渡します。

    pdf_out : Path
        出力される PDF ファイルのパスを渡します。
    '''
    # ファイルを読み込みます。
    dtc = np.loadtxt(data_dir / 'dtc.dat')
    eng = np.loadtxt(data_dir / 'eng.dat')
    men = np.loadtxt(data_dir / 'men.dat')

    t_dtc, dt, dt_lim, dt_N = dtc.T
    t_eng, eng_total = eng[:, 0], eng[:, 1]

    # men はオプション次第で描画します。
    t_men: Optional[np.ndarray] = None
    men_total: Optional[np.ndarray] = None
    if show_men:
        t_men, men_total = men[:, 0], men[:, 1]

    # fig. の設定を行います。
    nrows = 3 if show_men else 2
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(PAGE_W_IN, PAGE_H_IN),
        sharex=True
    )
    fig.subplots_adjust(
        left = LEFT / PAGE_W_IN, right = 1 - RIGHT / PAGE_W_IN,
        top = 1 - TOP / PAGE_H_IN, bottom = BOTTOM / PAGE_H_IN,
        hspace = HSPACE
    )

    # Δt のグラフを描画します。
    ax = axes[0]
    ax.plot(t_dtc, dt,      lw=0.5, label=r'$\Delta t$')
    ax.plot(t_dtc, dt_lim,  lw=0.5, label=r'$\Delta t_{\mathrm{limit}}$')
    ax.plot(t_dtc, dt_N,    lw=0.5, label=r'$\Delta t_N$')
    semilogy_formatter(ax)
    ax.set_ylabel(r'Time step size $\Delta t\,v_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_xlabel(r'Time $t\,v_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.tick_params(labelsize=7, direction='in')
    ax.tick_params(axis='x', labelbottom=True)
    ax.legend(fontsize=7, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Electostatic potential を描画します。
    ax = axes[1]
    ax.plot(t_eng, eng_total, lw=0.8, label='Total')
    for i in range(0, global_ny + 1):
        ax.plot(t_eng, eng[:, i+2], lw=0.5, label=rf'$m_y={i}$')
    semilogy_formatter(ax)
    ax.set_ylabel(r'Electrostatic potential $\langle\!|\varphi_k|^2\rangle\ [\delta^2T_{\mathrm{ref}}^2/e^2]$', fontsize=9)
    ax.set_xlabel(r'Time $t\,v_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.tick_params(labelsize=7, direction='in')
    ax.tick_params(axis='x', labelbottom=True)
    ax.legend(fontsize=7, ncol=2, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Vector potential を描画します。
    if show_men:
        ax = axes[2]
        ax.plot(t_men, men_total, lw=0.8, label='Total')
        for i in range(0, global_ny + 1):
            ax.plot(t_men, men[:, i+2], lw=0.5, label=rf'$m_y={i}$')
        semilogy_formatter(ax)
        ax.set_ylabel(r'Vector potential $\delta^2\langle|A_{\parallel k}|^2\rangle\ [\delta^2\rho_{\mathrm{ref}}^2B^2]$', fontsize=9)
        ax.set_xlabel(r'Time $t\,v_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
        ax.tick_params(labelsize=7, direction='in')
        ax.tick_params(axis='x', labelbottom=True)
        ax.legend(fontsize=7, ncol=2, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))

    # PDF ファイルを保存します。
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_out) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
