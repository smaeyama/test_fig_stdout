'''
calc_entropybalance.awk を Python に移植したスクリプトです。
'''
from pathlib import Path

import numpy as np
import pandas as pd

def _calc_entropy_balance(df: pd.DataFrame, non_uniform: bool = True) -> pd.DataFrame:
    '''
    GKV 出力の bln データから、時間微分を含むエントロピー バランスを計算します。

    Parameters
    ----------
    df : pd.DataFrame
        21 列のデータを持つ DataFrame (1 列目は時間) を渡します。

    non_uniform : bool, optional
        時間幅が等間隔でない場合に True を設定します。デフォルトは True です。

    Returns
    -------
    pd.DataFrame
        計算結果を含む新しい DataFrame (NaNを含む) を返します。
    '''
    df = df.copy()
    df.columns = [
        't',
        'Ss_nz', 'Ss_zf', 'WE_nz', 'WE_zf', 'WM_nz', 'WM_zf',
        'RE_nz', 'RE_zf', 'RM_nz', 'RM_zf',
        'NE_nz', 'NE_zf', 'NM_nz', 'NM_zf',
        'Ds_nz', 'Ds_zf',
        'GE', 'GM', 'QE', 'QM'
    ]

    def uniform_derivative(t, y):
        '''
        等間隔な時間ステップの場合に用いられる関数です。

        Parameters
        ----------
        t : Sequence[float]
            等間隔な時間の配列を渡します。

        y : Sequence[float]
            関数値を渡します。

        Returns
        -------
        NDArray[np.float64]
            1 次微分配列を返します。
        '''
        n = len(t)
        dydt = np.full(n, np.nan)
        for i in range(2, n - 2):
            dt = t[i+1] - t[i]
            cef = 1.0 / (12.0 * dt)

            y_vals = y[i-2:i+3]
            dydt[i] = cef * (
                -y_vals[4] + 8*y_vals[3] - 8*y_vals[1] + y_vals[0]
            )
        return dydt

    def non_uniform_derivative(t, y):
        '''
        等間隔でない時間ステップの場合に用いられる関数です。

        Parameters
        ----------
        t : Sequence[float]
            等間隔でない時間の配列を渡します。

        y : Sequence[float]
            関数値を渡します。

        Returns
        -------
        NDArray[np.float64]
            1 次微分配列を返します。
        '''
        n = len(t)
        dydt = np.full(n, np.nan)
        for ir in range(2, n - 2):
            t_m2, t_m1, t_0, t_p1, t_p2 = t[ir-2:ir+3]

            num_cefm2 = (t_0 - t_m1) * (t_0 - t_p1) * (t_0 - t_p2)
            den_cefm2 = (t_m2 - t_m1) * (t_m2 - t_0) * (t_m2 - t_p1) * (t_m2 - t_p2)
            cefm2 = num_cefm2 / den_cefm2

            num_cefm1 = (t_0 - t_m2) * (t_0 - t_p1) * (t_0 - t_p2)
            den_cefm1 = (t_m1 - t_m2) * (t_m1 - t_0) * (t_m1 - t_p1) * (t_m1 - t_p2)
            cefm1 = num_cefm1 / den_cefm1

            term1 = (t_0 - t_m1) * (t_0 - t_p1) * (t_0 - t_p2)
            term2 = (t_0 - t_m2) * (t_0 - t_p1) * (t_0 - t_p2)
            term3 = (t_0 - t_m2) * (t_0 - t_m1) * (t_0 - t_p2)
            term4 = (t_0 - t_m2) * (t_0 - t_m1) * (t_0 - t_p1)
            num_cefp0 = term1 + term2 + term3 + term4
            den_cefp0 = (t_0 - t_m2) * (t_0 - t_m1) * (t_0 - t_p1) * (t_0 - t_p2)
            cefp0 = num_cefp0 / den_cefp0

            num_cefp1 = (t_0 - t_m2) * (t_0 - t_m1) * (t_0 - t_p2)
            den_cefp1 = (t_p1 - t_m2) * (t_p1 - t_m1) * (t_p1 - t_0) * (t_p1 - t_p2)
            cefp1 = num_cefp1 / den_cefp1

            num_cefp2 = (t_0 - t_m2) * (t_0 - t_m1) * (t_0 - t_p1)
            den_cefp2 = (t_p2 - t_m2) * (t_p2 - t_m1) * (t_p2 - t_0) * (t_p2 - t_p1)
            cefp2 = num_cefp2 / den_cefp2

            dydt[ir] = (
                cefm2 * y[ir - 2] +
                cefm1 * y[ir - 1] +
                cefp0 * y[ir] +
                cefp1 * y[ir + 1] +
                cefp2 * y[ir + 2]
            )
        return dydt

    if non_uniform:
        df['dSsdt_nz'] = non_uniform_derivative(df['t'].values, df['Ss_nz'].values)
        df['dSsdt_zf'] = non_uniform_derivative(df['t'].values, df['Ss_zf'].values)
        df['dWEdt_nz'] = non_uniform_derivative(df['t'].values, df['WE_nz'].values)
        df['dWEdt_zf'] = non_uniform_derivative(df['t'].values, df['WE_zf'].values)
        df['dWMdt_nz'] = non_uniform_derivative(df['t'].values, df['WM_nz'].values)
        df['dWMdt_zf'] = non_uniform_derivative(df['t'].values, df['WM_zf'].values)
    else:
        df['dSsdt_nz'] = uniform_derivative(df['t'].values, df['Ss_nz'].values)
        df['dSsdt_zf'] = uniform_derivative(df['t'].values, df['Ss_zf'].values)
        df['dWEdt_nz'] = uniform_derivative(df['t'].values, df['WE_nz'].values)
        df['dWEdt_zf'] = uniform_derivative(df['t'].values, df['WE_zf'].values)
        df['dWMdt_nz'] = uniform_derivative(df['t'].values, df['WM_nz'].values)
        df['dWMdt_zf'] = uniform_derivative(df['t'].values, df['WM_zf'].values)

    return df

def _save_entropy_balance(df: pd.DataFrame, filepath: Path) -> None:
    '''
    計算済みの DataFrame を AWK スクリプトの出力と同等の形式でテキスト出力する関数です。

    Parameters
    ----------
    df : pd.DataFrame
        calc_entropy_balance 関数で処理された DataFrame を渡します。

    filepath : Path
        出力ファイル パスを渡します。
    '''
    with filepath.open('w') as f:
        # ヘッダー
        header_cols = [
            '#            time', 'dSsdt_nz', 'dSsdt_zf',
            'dWEdt_nz', 'dWEdt_zf', 'dWMdt_nz', 'dWMdt_zf',
            'RE_nz', 'RE_zf', 'RM_nz', 'RM_zf',
            'NE_nz', 'NE_zf', 'NM_nz', 'NM_zf',
            'Ds_nz', 'Ds_zf', 'GE', 'GM', 'QE', 'QM'
        ]
        f.write(''.join(f'{col:>17s}' for col in header_cols) + '\n')

        for i in range(len(df)):
            row = df.iloc[i]
            if i < 2 or i >= len(df) - 2:
                # NaN 部分は文字列として出力します。
                values = [row['t']] + ['NaN'] * 6 + [
                    row['RE_nz'], row['RE_zf'], row['RM_nz'], row['RM_zf'],
                    row['NE_nz'], row['NE_zf'], row['NM_nz'], row['NM_zf'],
                    row['Ds_nz'], row['Ds_zf'], row['GE'], row['GM'], row['QE'], row['QM']
                ]
            else:
                values = [
                    row['t'], row['dSsdt_nz'], row['dSsdt_zf'],
                    row['dWEdt_nz'], row['dWEdt_zf'], row['dWMdt_nz'], row['dWMdt_zf'],
                    row['RE_nz'], row['RE_zf'], row['RM_nz'], row['RM_zf'],
                    row['NE_nz'], row['NE_zf'], row['NM_nz'], row['NM_zf'],
                    row['Ds_nz'], row['Ds_zf'], row['GE'], row['GM'], row['QE'], row['QM']
                ]

            # 書式に従って書き出します。
            f.write(''.join(
                f'{v:17.7e}' if isinstance(v, (float, int)) and not isinstance(v, str) else f'{v:>17s}'
                for v in values) + '\n')

def save_entropy_balance(df: pd.DataFrame, filepath: Path, non_uniform: bool = True) -> None:
    '''
    GKV 出力の bln データから、時間微分を含むエントロピー バランスを計算し、
    テキスト形式で出力する関数です。

    Parameters
    ----------
    df : pd.DataFrame
        21 列のデータを持つ DataFrame (1 列目は時間) を渡します。

    filepath : Path
        出力ファイル パスを渡します。

    non_uniform : bool, optional
        時間幅が等間隔でない場合に True を設定します。デフォルトは True です。
    '''
    result = _calc_entropy_balance(df, non_uniform)
    _save_entropy_balance(result, filepath)
