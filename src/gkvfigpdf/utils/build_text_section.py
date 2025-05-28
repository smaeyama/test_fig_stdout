import datetime
import re
from pathlib import Path
from typing import List, Sequence, cast

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Preformatted, Spacer)

# 抽出対象パラメーター用の正規表現グループです。各タプルが 1 ブロック扱いになります。
_PATTERN_BLOCKS: Sequence[Sequence[str]] = [
    (
        r'nxw, nyw\s*=',
        r'global_ny\s*=',
        r'global_nz\s*=',
        r'global_nv, global_nm\s*=',
        r'nx, ny, nz\s*=',
        r'nv, nm\s*=',
        r'nzb, nvb\s*=',
        r'number of species\s*=',
        r'nproc',
    ),
    (
        r'q_0\s*=',
        r's_hat\s*=',
        r'eps_r\s*=',
        r's_input, s_0\s*=',
        r'nss, ntheta\s*=',
    ),
    (
        r'lx, ly, lz\s*=',
        r'lz,\s*z0\s*=',
        r'lz_l, z0_l\s*=',
        r'kxmin, kymin\s*=',
        r'kxmax, kymax\s*=',
        r'kperp_max\s*=',
        r'm_j, del_c\s*=',
        r'dz\s*=',
        r'dv, vmax\s*=',
        r'dm, mmax\s*=',
    ),
    (
        r'time_advnc\s*=',
        r'flag_time_adv\s*=',
        r'courant num\.',
        r'dt_perp\s*=',
        r'dt_zz\s*=',
        r'dt_vl\s*=',
        r'dt_col\s*=',
        r'dt_linear\s*=',
        r'dt_max\s*=',
        r'dt\s*=',
    ),
    (
        r'a, b, nu.*_ab\s*=',
    ),
]

def _namelist_to_flowables(namelist_path: Path,
                        paragraph_style: ParagraphStyle,
                        section_style: ParagraphStyle) -> List:
    '''
    namelist ファイルを ReportLab Flowable のリストへ変換する関数です。
    行頭 「&label」 を〈下線付き段落〉として強調し、以降の続きは Preformatted で保持します。
    &end 行はスキップし、行末に残る「, &end」も除去します。

    Parameters
    ----------
    namelist_path : Path
        gkvp_namelist.001 などの namelist ファイルのパスを渡します。

    paragraph_style : ParagraphStyle
        本文のスタイルを渡します。

    section_style : ParagraphStyle
        セクション見出し用のスタイルを渡します。

    Returns
    -------
    list[Flowable]
        ReportLab の Flowable を順序通り格納したリストです。
    '''
    flows: List = []
    section_rgx = re.compile(r'&([A-Za-z0-9_]+)', re.I)
    end_rgx = re.compile(r'^\s*&end', re.I)

    with namelist_path.open(encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            line = raw.rstrip('\n')

            # &end 行はスキップします。
            if end_rgx.match(line):
                continue

            # 行末の ", &end" / "&end" を削除します。
            line = re.sub(r',?\s*&end.*$', '', line)

            # 行頭 1 文字 + 空白 (例: "o ") を削除します。
            line = re.sub(r'^[A-Za-z]\s+', '', line)
            line_stripped = line.lstrip()

            m = section_rgx.match(line_stripped)
            if m and line_stripped.startswith('&'):
                name = m.group(1)
                # セクション開始行は下線を付けます。
                flows.append(Paragraph(f'<u>{name}</u>', section_style))

                remainder = line_stripped[m.end():].strip()
                if remainder:
                    flows.append(Preformatted(remainder, paragraph_style))
            else:
                # 通常行を出力します。
                flows.append(Preformatted(line_stripped, paragraph_style))
    return flows

def _extract_log_blocks(log_path: Path) -> List[List[str]]:
    '''
    _PATTERN_BLOCKS に従い、ログファイルから対応する最初の行を
    抽出してブロック毎にまとめる関数です。

    Parameters
    ----------
    log_path : Path
        ログ ファイルへのパスを渡します。

    Returns
    -------
    list[list[str]]
        外側リストがブロック、内側リストが抽出行のリストを返します。
        パターンに一致しなかった場合は空リストを要素に保持します。
    '''
    lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    blocks: List[List[str]] = []

    for pattern_group in _PATTERN_BLOCKS:
        group_lines: List[str] = []
        regexes = [re.compile(r'\s*' + p) for p in pattern_group]
        for rgx in regexes:
            for ln in lines:
                if rgx.search(ln):
                    group_lines.append(ln.strip())
                    break
        blocks.append(group_lines)

    return blocks

def _blocks_to_flowables(blocks: List[List[str]],
                        paragraph_style: ParagraphStyle) -> List:
    '''
    ログ抽出結果の 2 次元リストを ReportLab Flowable 化する関数です。
    ブロック間には 6 pt の垂直スペースを挿入します。

    Parameters
    ----------
    blocks : list[list[str]]
        _extract_log_blocks の戻り値を渡します。

    paragraph_style : ParagraphStyle
        本文用のスタイルを渡します。

    Returns
    -------
    list[Flowable]
        Flowable 化されたログ行を順序通りに並べたリストを返します。
    '''
    flows: List = []
    for bi, blk in enumerate(blocks):
        if bi > 0:
            flows.append(Spacer(0, 6))
        for ln in blk:
            flows.append(Preformatted(ln, paragraph_style))
    return flows

def build_text_section(namelist_file: Path, log_file: Path, pdf_out: Path,
                        margin_inch: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.6),
                        font_size: int = 10) -> None:
    '''
    出力 PDF ファイルのテキスト部を生成する関数です。

    Parameters
    ----------
    namelist_file : Path
        namelist ファイル (gkvp_namelist.001) のパスを渡します。

    log_file : Path
        ログ ファイル (gkvp.000000.0.log.001) のパスを渡します。

    pdf_out : Path
        出力 PDF のパスを渡します。

    margin_inch : tuple, optional
        ページの余白をインチ単位で渡します (left, right, top, bottom)。

    font_size : int, optional
        フォント サイズ [pt] です。
    '''
    left, right, top, bottom = margin_inch
    styles = getSampleStyleSheet()

    # 等幅のスタイルを設定します。
    mono = cast(ParagraphStyle, styles['Code'])
    mono.fontSize = font_size
    mono.leading  = font_size + 1

    # セクション用のスタイルを設定します。
    section_style = ParagraphStyle(
        'section', parent=styles['Code'],
        fontSize=font_size, leading=font_size+2,
        spaceBefore=4, spaceAfter=2
    )

    report: List = []
    # パラメーター部を設定します。
    dir_path = str(log_file.parent.parent)
    today = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report.append(Preformatted(dir_path,   mono))
    report.append(Preformatted(today, mono))
    report.append(Spacer(0, 4))
    report.extend(_namelist_to_flowables(namelist_file, mono, section_style))

    # ログ抽出部を設定します。
    report.append(Paragraph(f'<u>log</u>', section_style))
    blocks = _extract_log_blocks(log_file)
    report.extend(_blocks_to_flowables(blocks, mono))

    # PDF ファイルを生成します。
    doc = SimpleDocTemplate(
        str(pdf_out), pagesize = A4,
        leftMargin   = left * inch,
        rightMargin  = right * inch,
        topMargin    = top * inch,
        bottomMargin = bottom * inch
    )
    doc.build(report)
