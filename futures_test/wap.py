import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def wap(symbol, start_date: datetime, end_date: datetime, start_time: int, end_time: int):
    path = '/nas92/data/future'
    df = pd.DataFrame(columns=['symbol', 'date', 'start_time', 'end_time', 'twap', 'vwap'])

    if (start_time >= 210000000) | (start_time <= 23000000):
        file = f'/3s_quote(N)_{symbol}.parquet'
    else:
        file = f'/3s_quote_{symbol}.parquet'

    current_date = start_date
    while current_date <= end_date:
        current_date_str = current_date.strftime("%Y%m%d")
        sub_path = os.path.join(path, f'/{current_date_str}')
        df = pd.read_parquet(sub_path + file)
        df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        twap = df['last_prc'].mean()
        vwap = np.dot(df['last_prc'], df['t_volume']) / df['t_volume'].sum()
        temp_row = pd.Series({'symbol': symbol, 'date': current_date.date(), 'start_time': start_time,
                              'end_time': end_time, 'twap': twap, 'vwap': vwap})
        df = df.append(temp_row, ignore_index=True)
        current_date += timedelta(1)

    return df