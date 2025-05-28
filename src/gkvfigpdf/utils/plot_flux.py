from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
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
LEFT, RIGHT = 1.2, 1.8
TOP, BOTTOM = 1.2, 1.2
HSPACE, VSPACE = 0.4, 0.4

def _load_entropy(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    '''
    ent.<rank>.dat を読み込み、gnuplot スクリプトと同等の
    9 系列（dS/dt, R_sE, …, Error）を組み立てる関数です。

    Parameters
    ----------
    path : Path
        データ ファイルのパスを渡します。

    Returns
    -------
    t : NDArray[np.float64]
        1 次元の時間列を返します。

    series : NDArray[np.float64]
        形状 (Ntime, 9) のデータ行列を返します。

    labels : list[str]
        9 個の Matplotlib 用ラベルを返します。
    '''
    d = np.loadtxt(path)
    t = d[:, 0]
    series = np.column_stack([
        d[:, 1] + d[:, 2],          # dS_s/dt
        d[:, 7] + d[:, 8],          # R_{sE}
        d[:, 9] + d[:,10],          # R_{sM}
        d[:,17],                    # T_s Γ_{sE}/L_ps  (col 18)
        d[:,18],                    # T_s Γ_{sM}/L_ps  (col 19)
        d[:,19],                    # Theta_{sE}/L_Ts      (col 20)
        d[:,20],                    # Theta_{sM}/L_Ts      (col 21)
        d[:,15] + d[:,16],          # D_s              (16+17)
        (d[:, 1]+d[:, 2]) - (d[:, 7]+d[:, 8]) - (d[:, 9]+d[:,10])
        - (d[:,15]+d[:,16]) - d[:,17] - d[:,18] - d[:,19] - d[:,20]  # Error
    ])
    labels = [
        r'$\mathrm{d}S_s/\mathrm{d}t$', r'$R_{sE}$', r'$R_{sM}$',
        r'$T_s\Gamma_{sE}/L_{ps}$', r'$T_s\Gamma_{sM}/L_{ps}$',
        r'$\Theta_{sE}/L_{Ts}$', r'$\Theta_{sM}/L_{Ts}$',
        r'$D_s$', 'Error'
    ]
    return t, series, labels

def _load_flux(path: Path, ny: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    ges/gem/qes/qem.<rank>.dat 読み込み用の関数です。

    Parameters
    ----------
    path : Path
        データ ファイルのパスを渡します。

    ny : int
        global_ny の値を渡します。

    Returns
    -------
    dat[:, 0] : NDArray[np.float64]
        時間列を返します。

    dat[:, 1:ny+3] : NDArray[np.float64]
        形状 (Ntime, ny+2) のデータ行列を返します。
    '''
    dat = np.loadtxt(path)
    return dat[:, 0], dat[:, 1:ny+3]

def _plot_flux_panel(ax: Axes,
                    t: NDArray[np.float64],
                    arr: NDArray[np.float64],
                    title: str, ylabel: str) -> None:
    '''
    1 つのサブプロットのフラックス曲線を描画する関数です。

    Parameters
    ----------
    ax : Axes
        描画対象のサブプロットを渡します。

    t : NDArray[np.float64]
        時間軸を渡します。

    arr : NDArray[np.float64]
        (Ntime, ny+2) のフラックス値を渡します。

    title : str
        パネルのタイトルを渡します。

    ylabel : str
        y 軸ラベルを渡します。
    '''
    ax.set_title(title, fontsize=9)
    ax.plot(t, arr[:, 0], lw=0.8, label='Total')     # Total
    for i, col in enumerate(arr[:, 1:].T[::1]):
        ax.plot(t, col, lw=0.5, label=rf'$m_y={i}$')
    ax.set_xlabel(r'Time $tv_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=7, frameon=True, ncol=1,
                loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(labelsize=7, direction='in')
    ax.tick_params(axis='x', labelbottom=True)

    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.yaxis.get_offset_text().set_fontstyle('italic')

def plot_flux(rank: int, global_ny: int, data_dir: Path, pdf_out: Path) -> None:
    '''
    1 rank 分のフラックスの出力を A4 1 ページの PDF として出力する関数です。

    Parameters
    ----------
    rank : int
        対象ランク番号を渡します。

    global_ny : int
        m_y の最大番号を渡します。

    data_dir : Path
        *.dat を格納したデータ ディレクトリのパスを渡します。

    pdf_out : Path
        出力される PDF ファイルのパスを渡します。
    '''
    # データを読み込みます。
    t_ent, ent_arr, ent_labels = _load_entropy(data_dir/f'ent.{rank}.dat')
    t_ges, ges_arr = _load_flux(data_dir/f'ges.{rank}.dat', global_ny)
    t_gem, gem_arr = _load_flux(data_dir/f'gem.{rank}.dat', global_ny)
    t_qes, qes_arr = _load_flux(data_dir/f'qes.{rank}.dat', global_ny)
    t_qem, qem_arr = _load_flux(data_dir/f'qem.{rank}.dat', global_ny)

    # fig. の設定を行います。
    fig, axes = plt.subplots(
        3, 2, figsize=(PAGE_W_IN, PAGE_H_IN), sharex=True
    )
    fig.subplots_adjust(
        left = LEFT / PAGE_W_IN, right = 1 - RIGHT / PAGE_W_IN,
        top = 1 - TOP / PAGE_H_IN, bottom = BOTTOM / PAGE_H_IN,
        wspace = HSPACE, hspace = VSPACE
    )
    # 右上を空欄にします。
    axes[0,1].axis('off')

    # エントロピーを描画します。
    ax = axes[0,0]
    for col, lab in zip(ent_arr.T, ent_labels):
        ax.plot(t_ent, col, lw=0.5, label=lab)
    ax.set_xlabel(r'Time $tv_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(r'Entropy variables [$\delta^2 n_{\mathrm{ref}}T_{\mathrm{ref}}v_{\mathrm{ref}}/L_{\mathrm{ref}}$]', fontsize=9)
    ax.set_title(f'ranks = {rank}', fontsize=9)
    ax.legend(fontsize=7, ncol=1, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(labelsize=7, direction='in')
    ax.tick_params(axis='x', labelbottom=True)

    # GES を描画します。
    _plot_flux_panel(
        axes[1,0], t_ges, ges_arr, f'ranks = {rank}',
        r'Particle flux by ExB flows $ \Gamma_{sE}\,[\delta^2 n_{\mathrm{ref}}v_{\mathrm{ref}}]$'
    )
    # GEM を描画します。
    _plot_flux_panel(
        axes[1,1], t_gem, gem_arr, f'ranks = {rank}',
        r'Particle flux by magnetic flutters $ \Gamma_{sM}\,[\delta^2 n_{\mathrm{ref}}v_{\mathrm{ref}}]$'
    )
    # QES を描画します。
    _plot_flux_panel(
        axes[2,0], t_qes, qes_arr, f'ranks = {rank}',
        r'Energy flux by ExB flows $\Theta_{sE}\,[\delta^2 n_{\mathrm{ref}}T_{\mathrm{ref}}v_{\mathrm{ref}}]$'
    )
    # QEM を描画します。
    _plot_flux_panel(
        axes[2,1], t_qem, qem_arr, f'ranks = {rank}',
        r'Energy flux by magnetic flutters $\Theta_{sM}\,[\delta^2 n_{\mathrm{ref}}T_{\mathrm{ref}}v_{\mathrm{ref}}]$'
    )

    # 凡例を削除して統合します。
    axes[1, 0].get_legend().remove()
    axes[2, 0].get_legend().remove()
    axes[2, 1].get_legend().remove()

    # PDF ファイルを保存します。
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_out) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
