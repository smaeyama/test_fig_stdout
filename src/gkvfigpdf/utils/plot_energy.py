from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes

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
TOP, BOTTOM = 1.8, 1.8
HSPACE, VSPACE = 0.4, 0.4

def load_ent_for_ranks(data_dir: Path, nprocs: int
                        ) -> tuple[np.ndarray,               # t
                                    list[np.ndarray],        # series_e
                                    list[str],               # labels_e
                                    list[np.ndarray],        # series_m
                                    list[str]]:              # labels_m
    '''
    すべての ent.<rank>.dat ファイルを読み込み、エネルギー収支項を抽出する関数です。

    Parameters
    ----------
    data_dir : Path
        dat ファイルが格納されたディレクトリのパスを渡します。

    nprocs : int
        プロセス数を渡します。

    Returns
    -------
    t
        共通の時間軸 (rank0 の 1 列目) を返します。

    series_e, labels_e
        dW_E/dt と -R_{sE} の各 rank の系列とラベルを返します。

    series_m, labels_m
        dW_M/dt と -R_{sM} の各 rank の系列とラベルを返します。
    '''
    # rank 0 を読み込みます。
    ent0 = np.loadtxt(data_dir/'ent.0.dat')
    t = ent0[:, 0]

    # 電場エネルギー
    dw_e = ent0[:, 3] + ent0[:, 4]
    series_e = [dw_e]
    labels_e = [r'$\mathrm{d}W_E/\mathrm{d}t$']

    # 磁場エネルギー
    dw_m = ent0[:, 5] + ent0[:, 6]
    series_m = [dw_m]
    labels_m = [r'$\mathrm{d}W_M/\mathrm{d}t$']

    # 各 rank の R_sE、R_sM を取得します。
    for r in range(nprocs):
        path = data_dir/f'ent.{r}.dat'
        if not path.exists():
            continue
        d = np.loadtxt(path)
        rsE = -(d[:, 7] + d[:, 8])
        rsM = -(d[:, 9] + d[:,10])
        series_e.append(rsE);  labels_e.append(rf'$-R_{{sE}}(s={r})$')
        series_m.append(rsM);  labels_m.append(rf'$-R_{{sM}}(s={r})$')

    return t, series_e, labels_e, series_m, labels_m

def load_energy(path: Path, ny: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    wes.dat / wem.dat ファイルを読み込む関数です。

    Parameters
    ----------
    path : Path
        読み込むデータ ファイルのパスを渡します。

    ny : int
        global_ny の値を渡します。

    Returns
    -------
    pandas.DataFrame
        ['label', 'value'] 列を持つ DataFrame を返します。
    '''
    d = np.loadtxt(path)
    t = d[:, 0]
    arr = d[:, 1:ny+3]      # shape = (Ntime, ny+2)
    return t, arr

def _plot_energy(ax: Axes, t: NDArray[np.float64], arr: NDArray[np.float64], ylabel: str) -> None:
    '''
    wes / wem のサブプロット描画用の関数です。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        描画対象のサブプロットを渡します。
    t : NDArray[np.float64]
        1 次元の時間軸データを渡します。
    arr : NDArray[np.float64]
        2 次元配列 (Ntime, Nseries) を渡します。
        col 0 が Total、以降は m_y 成分です。
    ylabel : str
        y 軸ラベルを渡します。
    '''
    ax.plot(t, arr[:, 0], lw=0.8, label='Total')
    for i, col in enumerate(arr[:, 1:].T[::1]):
        ax.plot(t, col, lw=.5, label=rf'$m_y={i}$')
    ax.set_yscale('log');   ax.yaxis.set_major_formatter(LogFormatter(10))
    ax.set_xlabel(r'Time $tv_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=7, ncol=2, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(labelsize=7, direction='in')
    ax.tick_params(axis='x', labelbottom=True)

def plot_energy(nprocs: int, global_ny: int, data_dir: Path, pdf_out: Path) -> None:
    '''
    ent.*.dat / wes.dat / wem.dat を用いてエネルギー図を PDF 出力する関数です。

    Parameters
    ----------
    nprocs : int
        ent.*.dat の rank 数を渡します。

    global_ny : int
        wes / wem の最大 m_y 番号を渡します。

    data_dir : Path
        データ ファイル *.dat が格納されたディレクトリ パスを渡します。

    pdf_out : Path
        出力 PDF ファイルのパスを渡します。
    '''
    # ent.*.dat から dW/dt と -R_s を取得します。
    (t_ent, series_e, labels_e,
    series_m, labels_m) = load_ent_for_ranks(data_dir, nprocs)

    # wes.dat を読み込みます。
    t_wes, wes_arr = load_energy(data_dir/'wes.dat', global_ny)

    # fig. の設定を行います。
    fig, axes = plt.subplots(
        2, 2, figsize=(PAGE_W_IN, PAGE_H_IN), sharex='col'
    )
    fig.subplots_adjust(
        left = LEFT / PAGE_W_IN, right = 1 - RIGHT / PAGE_W_IN,
        top = 1 - TOP / PAGE_H_IN, bottom = BOTTOM / PAGE_H_IN,
        wspace = HSPACE, hspace = VSPACE
    )

    # (0, 0): dW_E/dt および -R_sE を描画します。
    ax = axes[0,0]
    for s, lab in zip(series_e, labels_e):
        ax.plot(t_ent, s, lw=0.5, label=lab)
    ax.set_xlabel(r'Time $tv_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(r'Entropy variables [$\delta^2 n_{\mathrm{ref}}T_{\mathrm{ref}}v_{\mathrm{ref}}/L_{\mathrm{ref}}$]', fontsize=9)
    ax.legend(fontsize=7, ncol=1, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(labelsize=7, direction='in')
    ax.tick_params(axis='x', labelbottom=True)

    # (0, 1): dW_M/dt および -R_sM を描画します。
    ax = axes[0,1]
    for s, lab in zip(series_m, labels_m):
        ax.plot(t_ent, s, lw=0.5, label=lab)
    ax.set_xlabel(r'Time $tv_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(r'Entropy variables [$\delta^2 n_{\mathrm{ref}}T_{\mathrm{ref}}v_{\mathrm{ref}}/L_{\mathrm{ref}}$]', fontsize=9)
    ax.legend(fontsize=7, ncol=1, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(labelsize=7, direction='in')
    ax.tick_params(axis='x', labelbottom=True)

    # (1, 0): W_E (対数軸) を描画します。
    _plot_energy(
        axes[1,0], t_wes, wes_arr,
        r'Electrostatic energy $W_E\,[\delta^2 n_{\mathrm{ref}}T_{\mathrm{ref}}]$'
    )

    # 上段パネル (0,0) の凡例を削除し、凡例を 1 箇所に集約します。
    axes[0, 0].get_legend().remove()

    # (1, 1): nprocs の値に応じて W_M または空欄を描画します。
    if nprocs > 1:
        t_wem, wem_arr = load_energy(data_dir/'wem.dat', global_ny)
        _plot_energy(
            axes[1,1], t_wem, wem_arr,
            r'Magnetic field energy $W_M\,[\delta^2 n_{\mathrm{ref}}T_{\mathrm{ref}}]$'
        )
        # 左下パネルの凡例を削除します。
        axes[1, 0].get_legend().remove()
    else:
        # パネルを空欄にします。
        axes[1, 1].axis('off')

    # PDF ファイルに保存します。
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_out) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
