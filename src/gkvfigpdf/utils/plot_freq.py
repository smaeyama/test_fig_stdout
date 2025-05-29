from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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
LEFT, RIGHT = 1.2, 1.2
TOP, BOTTOM = 2.2, 2.2
HSPACE, VSPACE = 0.4, 0.2

def plot_freq(global_ny: int, data_dir: Path, pdf_out: Path) -> None:
    '''
    frq.dat と dsp.dat をもとに成長率と周波数を描画し、PDF を出力する関数です。

    Parameters
    ----------
    global_ny : int
        m_y の最大番号を渡します。frq.dat に含まれる列数の上限に対応します。

    data_dir : Path
        frq.dat、dsp.dat が置かれたディレクトリ パスを渡します。

    pdf_out : Path
        出力される PDF ファイルのパスを渡します。
    '''
    # frq.dat の読み込みと時系列後半の抽出を行います。
    frq = np.loadtxt(data_dir / 'frq.dat')
    t   = frq[:, 0]
    tend = t[-1]
    mask = t >= tend / 2
    t_cut = t[mask]
    ncols = frq.shape[1]

    # fig. の設定を行います。
    fig, axes = plt.subplots(
        2, 2, figsize=(PAGE_W_IN, PAGE_H_IN), sharex='row'
    )
    fig.subplots_adjust(
        left = LEFT / PAGE_W_IN, right = 1 - RIGHT / PAGE_W_IN,
        top = 1 - TOP / PAGE_H_IN, bottom = BOTTOM / PAGE_H_IN,
        wspace = HSPACE, hspace = VSPACE
    )

    # (0, 0): Growthrate γ_l(t) を描画します。
    ax = axes[0, 0]
    for my in range(1, global_ny + 1):
        col_idx = 2 * my
        if col_idx >= ncols:
            break
        ax.plot(t_cut, frq[mask, col_idx],
                lw=0.5, label=rf'$m_y={my}$')
    ax.set_xlabel(r'Time $tv_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(r'Growthrate $\gamma_\ell\,[v_{\mathrm{ref}}/L_{\mathrm{ref}}]$', fontsize=9)
    ax.legend(fontsize=6, ncol=1, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(labelsize=7, direction='in')

    # (0, 1): Frequency ω_r(t) を描画します。
    ax = axes[0, 1]
    for my in range(1, global_ny + 1):
        col_idx = 2 * my - 1
        if col_idx >= ncols:
            break
        ax.plot(t_cut, frq[mask, col_idx],
                lw=0.5, label=rf'$m_y={my}$')
    ax.set_xlabel(r'Time $tv_{\mathrm{ref}}/L_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(r'Frequency $\omega_r\,[v_{\mathrm{ref}}/L_{\mathrm{ref}}]$', fontsize=9)
    ax.legend(fontsize=6, ncol=1, frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(labelsize=7, direction='in')

    # dsp.dat から k_y スペクトルを抽出します (k_x ≈ 0 の行)。
    try:
        dsp = np.loadtxt(data_dir / 'dsp.dat')
        if dsp.ndim == 1:
            dsp = dsp.reshape(1,-1)
        ky_mask = np.abs(dsp[:, 0]) < 1e-10      # k_x ≈ 0
        ky  = dsp[ky_mask, 1]                    # col2
        freq = dsp[ky_mask, 2]                   # col3
        grow = dsp[ky_mask, 3]                   # col4
    
    except (OSError, ValueError, IndexError) as e:
        print(f"[WARN] dsp.dat is not converged.")
        ky = np.array([])
        freq = np.array([])
        grow = np.array([])

    # (1, 0): γ_l(k_y) 成長率スペクトルを描画します。
    ax = axes[1, 0]
    ax.plot(ky, grow, lw=0.8, marker='+', markersize=4)
    ax.set_xlabel(r'Poloidal wave number $k_y\rho_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(r'Growthrate $\gamma_\ell\,[v_{\mathrm{ref}}/L_{\mathrm{ref}}]$', fontsize=9)
    ax.tick_params(labelsize=7, direction='in')

    # (1, 1): ω_r(k_y) 周波数スペクトルを描画します。
    ax = axes[1, 1]
    ax.plot(ky, freq, lw=0.8, marker='+', markersize=4)
    ax.set_xlabel(r'Poloidal wave number $k_y\rho_{\mathrm{ref}}$', fontsize=9)
    ax.set_ylabel(r'Frequency $\omega_r\,[v_{\mathrm{ref}}/L_{\mathrm{ref}}]$', fontsize=9)
    ax.tick_params(labelsize=7, direction='in')

    # 上段左パネルの凡例を削除します。
    axes[0, 0].get_legend().remove()

    # PDF ファイルを保存します。
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_out) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
