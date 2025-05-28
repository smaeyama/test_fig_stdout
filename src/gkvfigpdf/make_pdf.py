import shutil
import argparse
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
from pypdf import PdfReader, PdfWriter, PageObject
from reportlab.pdfgen import canvas
from datetime import datetime

from .utils.parse_parameter_setting import parse_parameters
from .utils.calc_elt import calc_elt
from .utils.calc_entropy_balance import save_entropy_balance
from .utils.build_text_section import build_text_section
from .utils.plot_elt import plot_elt
from .utils.plot_mtrf import plot_mtr, plot_mtf
from .utils.plot_freq import plot_freq
from .utils.plot_time_series import plot_time_series
from .utils.plot_flux import plot_flux
from .utils.plot_energy import plot_energy

def clean_directory(dir_path: Path) -> None:
    '''
    指定されたパスに対応するディレクトリが存在する場合は、
    その内容 (ファイルおよびサブディレクトリ) をすべて削除します。
    ディレクトリが存在しない場合は、新たに作成します。

    Parameters
    ----------
    dir_path : Path
        対象とするディレクトリ パスを指定します。
    '''
    if dir_path.exists():
        if dir_path.is_dir():
            for item in dir_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            raise ValueError(f'{dir_path} はディレクトリではありません。')
    else:
        dir_path.mkdir(parents=True)

def concat_files(src_dir: Path, pattern: str, dst_path: Path) -> None:
    '''
    パターンに一致する複数のファイルを連結し、1 つのファイルに結合して保存します。

    Parameters
    ----------
    src_dir : Path
        元ファイルが存在するディレクトリ パスを指定します。

    pattern : str
        結合対象とするファイルのパターンを指定します。

    dst_path : Path
        出力ファイルのパスを指定します。
    '''
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open('wb') as outfile:
        # ファイル名で昇順にソートして結合します。
        for src_file in sorted(src_dir.glob(pattern)):
            with src_file.open('rb') as infile:
                outfile.write(infile.read())

def page_number_overlay(page_width: float,
                        page_height: float,
                        text: str,
                        font_name: str = 'Helvetica',
                        font_size: int = 9) -> PageObject:
    '''
    ページ番号だけを書き込んだ透明な PDF ページを生成し、
    pypdf の PageObject として返す関数です。

    Parameters
    ----------
    page_width : float
        ページの幅 [pt] です。元ページの mediabox.width をそのまま渡します。

    page_height : float
        ページの高さ [pt] です。元ページの mediabox.height をそのまま渡します。

    text : str
        表示するページ番号文字列を渡します。

    font_name : str, optional
        フォント名です。デフォルトは Helvetica です。

    font_size : int, optional
        フォント サイズ [pt] です。デフォルトは 9 pt です。

    Returns
    -------
    PageObject
        ページ番号が書かれた透明ページを返します。
    '''
    buf = BytesIO()
    can = canvas.Canvas(buf, pagesize=(page_width, page_height))
    can.setFont(font_name, font_size)
    can.drawCentredString(page_width / 2, 20, text)
    can.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]

def merge_pdfs(pdf_paths: Iterable[Path], out_pdf: Path) -> None:
    '''
    複数 PDF を順番通りに結合し、各ページに
    ページ番号を付けて保存する関数です。

    Parameters
    ----------
    pdf_paths : Iterable[Path]
        結合対象 PDF のパスを並べたイテラブル オブジェクトを渡します。
        与えた順序の通りにページが連結されます。

    out_pdf : Path
        結果を書き出す PDF ファイル パスです。
    '''
    writer = PdfWriter()

    # すべてのページを読み込み、リストに登録します。
    pages: list[PageObject] = []
    for path in pdf_paths:
        reader = PdfReader(path)
        pages.extend(reader.pages)

    total = len(pages)

    # ページ番号をオーバレイします。
    for i, page in enumerate(pages, start=1):
        w, h = page.mediabox.width, page.mediabox.height
        overlay = page_number_overlay(w, h, f'{i} / {total}')
        page.merge_page(overlay)
        writer.add_page(page)

    # ファイルを保存します。
    with out_pdf.open('wb') as f:
        writer.write(f)

def gkvfigpdf(gkv_stdout_dir: Union[str, Path]):
    """
    Generate a GKV figure summary PDF from a log directory.

    Parameters
    ----------
    gkv_stdout_dir : str or Path
        Path to the directory containing GKV output files.

    Raises
    ------
    FileNotFoundError
        If required directories or files do not exist.
    """
    gkv_stdout_dir = Path(gkv_stdout_dir).expanduser().resolve()

    # Validate required paths
    hst_dir = gkv_stdout_dir / 'hst'
    log_filepath = gkv_stdout_dir / 'log/gkvp.000000.0.log.001'
    namelist_filepath = gkv_stdout_dir / 'gkvp_namelist.001'

    if not gkv_stdout_dir.is_dir():
        raise FileNotFoundError(f'GKV standard output directory not found: {gkv_stdout_dir}')
    if not hst_dir.is_dir():
        raise FileNotFoundError(f'hst directory not found: {hst_dir}')
    if not log_filepath.is_file():
        raise FileNotFoundError(f'Log file not found: {log_filepath}')
    if not namelist_filepath.is_file():
        raise FileNotFoundError(f'Namelist file not found: {namelist_filepath}')

    # 実行ディレクトリを取得し、作業ディレクトリを設定します。
    cwd = Path.cwd()
    timestamp = datetime.now().strftime("figpdf_%Y%m%d_%H%M%S")
    out_root = cwd / timestamp
    data_dir = out_root / 'data'
    fig_dir  = out_root / 'fig'

    # 既存の出力ファイルを削除します。
    clean_directory(data_dir)
    clean_directory(fig_dir)

    # ログ ファイルから処理に必要なパラメーターを抽出します。
    nprocs, global_ny, calc_type = parse_parameters(log_filepath)

    # 可視化対象のデータ ファイルを作業ディレクトリへ配置します。
    calc_elt(log_filepath, data_dir)

    copy_filenames = ['mtr', 'mtf']
    for name in copy_filenames:
        shutil.copyfile(hst_dir / f'gkvp.{name}.001', data_dir / f'{name}.dat')

    concat_filenames = ['dtc', 'eng', 'men', 'wes', 'wem']
    for name in concat_filenames:
        concat_files(hst_dir, f'gkvp.{name}.*', data_dir / f'{name}.dat')

    concat_proc_filenames = ['ges', 'gem', 'qes', 'qem', 'bln']
    for i in range(nprocs):
        for name in concat_proc_filenames:
            concat_files(hst_dir, f'gkvp.{name}.{i}.*', data_dir / f'{name}.{i}.dat')
        df_bln = pd.read_csv(data_dir / f'bln.{i}.dat', sep=r'\s+', header=None)
        save_entropy_balance(df_bln, data_dir / f'ent.{i}.dat')

    if calc_type == 'lin_freq':
        concat_files(hst_dir, 'gkvp.frq.*', data_dir / 'frq.dat')
        # dsp ファイルのうち、ファイル サイズ > 0 の最後の 1 ファイルをコピーします。
        dsp_candidates = [f for f in sorted(hst_dir.glob('*.dsp.*')) if f.stat().st_size > 0]
        if dsp_candidates:
            last_dsp = dsp_candidates[-1]
            shutil.copyfile(last_dsp, data_dir / 'dsp.dat')

    # 出力された一時的な PDF ファイルをのパスのリストです。
    pdfs = []

    # テキスト出力をまとめた PDF を出力します。
    pdf_filepath = fig_dir / 'text_section.pdf'
    build_text_section(
        namelist_file = log_filepath.parent.parent / 'gkvp_namelist.001',
        log_file = log_filepath, pdf_out = pdf_filepath)
    pdfs.append(pdf_filepath)

    # 以下では 1 ページずつ fig を出力します。
    pdf_filepath = fig_dir / 'plot_elt.pdf'
    plot_elt(data_dir, pdf_filepath)
    pdfs.append(pdf_filepath)

    pdf_filepath = fig_dir / 'plot_mtr.pdf'
    plot_mtr(data_dir / 'mtr.dat', pdf_filepath)
    pdfs.append(pdf_filepath)

    pdf_filepath = fig_dir / 'plot_mtf.pdf'
    plot_mtf(data_dir / 'mtf.dat', pdf_filepath)
    pdfs.append(pdf_filepath)

    if calc_type == 'lin_freq':
        pdf_filepath = fig_dir / 'plot_freq.pdf'
        plot_freq(global_ny, data_dir, pdf_filepath)
        pdfs.append(pdf_filepath)

    pdf_filepath = fig_dir / 'plot_time_series.pdf'
    plot_time_series(global_ny, nprocs > 1, data_dir, pdf_filepath)
    pdfs.append(pdf_filepath)

    for rank in range(nprocs):
        pdf_filepath = fig_dir / f'plot_flux.{rank}.pdf'
        plot_flux(rank, global_ny, data_dir, pdf_filepath)
        pdfs.append(pdf_filepath)

    pdf_filepath = fig_dir / 'plot_energy.pdf'
    plot_energy(nprocs, global_ny, data_dir, pdf_filepath)
    pdfs.append(pdf_filepath)

    # リスト内の PDF ファイルを結合し、最終出力 PDF を作成します。
    pdf_filepath = out_root / 'fig_stdout.pdf'
    merge_pdfs(pdfs, pdf_filepath)
    print(f'{pdf_filepath} generated.')


def main() -> None:
    '''CLI entry'''

    parser = argparse.ArgumentParser(
        usage='python -m gkvfigpdf [-h] [-d DIR]',
        description='This script generates a PDF report from GKV simulation output.'
    )
    parser.add_argument(
        '-d', '--dir', '-dir', required=False, type=Path,
        help='Specify the path to the directory containing the GKV output files.'
    )
    
    args = parser.parse_args()
    gkv_stdout_dir: Optional[Path] = args.dir
    
    if gkv_stdout_dir is None:
        parser.print_help()
        parser.exit(
            status = 2,
            message = 'Error: Please specify the log directory using the --dir option.\n'
        )
    else:
        gkv_stdout_dir = gkv_stdout_dir.expanduser().resolve()
    
    try:
        gkvfigpdf(args.dir)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit(1)

if __name__ == '__main__':
    main()
