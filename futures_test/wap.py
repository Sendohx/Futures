import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def wap(symbol, start_date: int, end_date: int, start_time: int, end_time: int, base = '/nas92/data/future/'):
    result = pd.DataFrame(columns=['symbol', 'date', 'start_time', 'end_time', 'twap', 'vwap'])

    file1 = f'/3s_quote(N)_{symbol}.parquet'
    file2 = f'/3s_quote_{symbol}.parquet'

    current_date = start_date
    while current_date <= end_date:
        sub_path = os.path.join(base, str(current_date))
        if not os.path.exists(sub_path):
            print(f'no data in dir {sub_path}')
            current_date += 1
            continue

        if os.path.exists(sub_path + file1):
            df = pd.read_parquet(sub_path + file1)
        elif os.path.exists(sub_path + file2):
            df = pd.read_parquet(sub_path + file2)
        else:
            print(f'no data for {symbol} on {current_date}')
            current_date += 1
            continue

        if (start_time <= 240000000) & (end_time <= 23000000):
            mask1 = (df['time'] >= start_time) & (df['time'] <= 240000000)
            mask2 = (df['time'] >= 0) & (df['time'] < end_time)
            df.loc[mask1, 'time'] = df.loc[mask1, 'time'] - 120000000
            df.loc[mask2, 'time'] = df.loc[mask2, 'time'] + 120000000
            df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
            twap = df['last_prc'].mean()
            vwap = np.dot(df['last_prc'], df['t_volume']) / df['t_volume'].sum()
            temp_row = pd.Series({'symbol': symbol, 'date': current_date, 'start_time': start_time,
                                  'end_time': end_time, 'twap': twap, 'vwap': vwap})
            df = df.append(temp_row, ignore_index=True)
        else:
            df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
            twap = df['last_prc'].mean()
            vwap = np.dot(df['last_prc'], df['t_volume']) / df['t_volume'].sum()
            temp_row = pd.Series({'symbol': symbol, 'date': current_date, 'start_time': start_time,
                                  'end_time': end_time, 'twap': twap, 'vwap': vwap})
            result = result.append(temp_row, ignore_index=True)

        current_date += 1

    return result

if __name__ == '__main__':
    symbol = 'CU2302.SHF'
    start_date = 20230101
    end_date = 20230106
    start_time = 92000000
    end_time = 100000000

    df_wap = wap(symbol, start_date, end_date, start_time, end_time)
    print(df_wap)