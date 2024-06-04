import os
import pandas as pd
import numpy as np
from datetime import datetime


def pre_process_quote_day(df):
    """
    quote   9:00 - 15:00
    :param df:
    :return:
    """

    # 把int转换成时间序列数据类型
    def convert_int_to_time(time_int):
        time_str = str(int(time_int)).zfill(9)
        formatted_time_str = f'{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}.{time_str[6:]}'
        try:
            if len(formatted_time_str.split('.')) >= 3:
                formatted_time_str = formatted_time_str.split('.')[0] + '.' + formatted_time_str.split('.')[1]
            datetime_object = datetime.strptime(formatted_time_str, '%H:%M:%S.%f')
        except ValueError:
            print(f"{formatted_time_str} error")
            return None
        return datetime_object

    # 把时间序列数据转换成int类型
    def convert_time_to_int(time_obj):
        time_str = time_obj.strftime('%H%M%S%f')[:-3]
        return int(time_str)



    df = df[df['night_trading'] == 0]  # 日盘
    columns_to_keep = ['symbol', 'trading_date', 'date', 'time', 'recv_time', 'last_prc', 'open_interest', 'volume', 'turnover',
                       'ask_prc1', 'bid_prc1', 'ask_vol1', 'bid_vol1', 'acc_high', 'acc_low',
                       'high_limited', 'low_limited', 'prev_close', 'acc_open', 'night_trading']
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
    df.drop(columns=columns_to_drop, inplace=True)  # 去掉多余列
    df.sort_values(by='time', inplace=True)  # 按时间排序
    if 'recv_time' not in df.columns:
        df['recv_time'] = np.nan  # 加上receive time列

    if 90003000 not in df['time'].values:  # 加上时间在3点整的一行
        new_row = pd.DataFrame({
            'symbol': np.nan,
            'trading_date': np.nan,
            'date': np.nan,
            'time': [90003000],
            'recv_time': np.nan,
            'last_prc': np.nan,
            'open_interest': np.nan,
            'volume': np.nan,
            'turnover': np.nan,
            'ask_prc1': np.nan,
            'bid_prc1': np.nan,
            'ask_vol1': np.nan,
            'bid_vol1': np.nan,
            'acc_high': np.nan,
            'acc_low': np.nan,
            'high_limited': np.nan,
            'low_limited': np.nan,
            'prev_close': np.nan,
            'acc_open': np.nan,
            'night_trading': 0
        })

        df = pd.concat([df, new_row])
        df.sort_values(by='time', inplace=True)
        df.reset_index(drop=True, inplace=True)

        max_time_row = df[df['time'] <= 90003000]['time'].idxmax()  # 最大time值对应的行号
        closing_call_row = df[df['time'] == 90003000]['time'].idxmax()  # 3点对应的最大行号
        max_time_values = df.loc[max_time_row, :]  # 最大time值对应的行

        for col in df.columns:
            df.at[closing_call_row, col] = max_time_values[col]  # 把最大时间对应的行的数据放到3点对应的行
        df.at[closing_call_row, 'time'] = 90003000  # 并把该行时间写成150000000

    if 150000000 not in df['time'].values:  # 加上时间在3点整的一行
        new_row = pd.DataFrame({
            'symbol': np.nan,
            'trading_date': np.nan,
            'date': np.nan,
            'time': [150000000],
            'recv_time': np.nan,
            'last_prc': np.nan,
            'open_interest': np.nan,
            'volume': np.nan,
            'turnover': np.nan,
            'ask_prc1': np.nan,
            'bid_prc1': np.nan,
            'ask_vol1': np.nan,
            'bid_vol1': np.nan,
            'acc_high': np.nan,
            'acc_low': np.nan,
            'high_limited': np.nan,
            'low_limited': np.nan,
            'prev_close': np.nan,
            'acc_open': np.nan,
            'night_trading': 0
        })

        df = pd.concat([df, new_row])
        df.sort_values(by='time', inplace=True)
        df.reset_index(drop=True, inplace=True)

    max_time_row = df[df['time'] >= 150000000]['time'].idxmax()  # 最大time值对应的行号
    closing_call_row = df[df['time'] == 150000000]['time'].idxmax()  # 3点对应的最大行号
    max_time_values = df.loc[max_time_row, :]  # 最大time值对应的行

    for col in df.columns:
        df.at[closing_call_row, col] = max_time_values[col]  # 把最大时间对应的行的数据放到3点对应的行
    df.at[closing_call_row, 'time'] = 150000000  # 并把该行时间写成150000000

    df['time'] = df['time'].apply(convert_int_to_time)  # 转换成datetime
    df.dropna(subset=['time'], inplace=True)  # 去掉time空值的行

    if not (df.recv_time.isnull().all() or df.recv_time.isna().all()):
        df = df.dropna(subset=['recv_time'])  # 去掉receive_time空值的行
        idx_to_keep = df.groupby('time')['recv_time'].idxmin()  # receive时间并非全空值时，每个时间按receive的最小时间筛选
        df = df.loc[idx_to_keep].copy()
    else:
        idx_to_keep = df.groupby('time')['time'].idxmin()  # receive时间全空值时，每个时间按最小时间筛选
        df = df.loc[idx_to_keep].copy()

    df_resampled = df.set_index('time').resample('3S').asfreq()  # 时间设成索引，重采样，3s一个行情
    df_resampled.reset_index(inplace=True)
    df_resampled['time'] = df_resampled['time'].apply(convert_time_to_int)
    # df_resampled.set_index('time', inplace=True)
    df['time'] = df['time'].apply(convert_time_to_int)
    # df.set_index('time',inplace=True)
    pd.set_option('display.max_columns', None)  # 最大显示列数为无限
    df = pd.merge(df, df_resampled, how='outer')  # 按index合并，主要目的就是把每一个time对应一个重采样的时间
    df = df.sort_values('time')
    df.reset_index(drop=True, inplace=True)

    df['fill'] = df.apply(lambda row: 1 if row.isnull().any() else 0, axis=1)  # 对存在空值的行的fill列填1，否则0
    df['Time'] = df.apply(lambda x: x['time'] if x[['last_prc', 'volume']].notna().all() else np.nan,
                          axis=1)  # last_prc和volume不全为空时保留，否则nan

    last_non_na_rectime = None  # 上一个不是空值的receive time，用于填充下一个
    last_non_na_time = None  # 上一个不是空值的time，用于填充下一个

    for idx, row in df.iterrows():
        if pd.notna(row['recv_time']):
            last_non_na_rectime = row['recv_time']
        else:
            df.at[idx, 'recv_time'] = last_non_na_rectime

        if pd.notna(row['Time']):
            last_non_na_time = row['Time']
        else:
            df.at[idx, 'Time'] = last_non_na_time

    df = df.sort_values('time')
    df.ffill(inplace=True)

    df['time'] = df['time'].apply(convert_int_to_time)
    df = df.set_index('time').resample('3S').asfreq()
    df.ffill(inplace=True)
    df.reset_index(inplace=True)
    df['time'] = df['time'].apply(convert_time_to_int)

    index_location1 = df[df['time'] >= 90000000].index.min()  # 取对应索引
    if pd.isna(index_location1):  # Check if index_location1 is NaN
        pd.DataFrame()
    # try:
    #     row_number1 = df.index.get_loc(index_location1)  # 取行号，其实多此一举了，因为已经reset_index了
    # except KeyError:
    #     print(f"Index {index_location1} not found.")
    #     pd.DataFrame()

    index_location2 = df[df['time'] == 150000000].index.max()
    # row_number2 = df.index.get_loc(index_location2)
    df = df.loc[index_location1: index_location2].reset_index(drop=True)  # 取9-15区间内数据

    df = df.sort_values('time')
    start_index = df[df['time'] >= 90000000].index[0]
    df['t_volume'] = 0
    for idx in range(start_index, len(df)):
        if idx == 0:
            df.loc[idx, 't_volume'] = 0
        else:
            df.loc[idx, 't_volume'] = df.loc[idx, 'volume'] - df.loc[idx - 1, 'volume']  # 时刻的volume（原始数据为累计volume）
    df['t_volume'] = df['t_volume'].fillna(0)

    df = df.sort_values('time')
    df['middle'] = None  # 中间价
    df.loc[:start_index + 1, 'middle'] = 0
    for idx in range(start_index, len(df)):
        vol = df.loc[idx, 't_volume']
        bid = df.loc[idx, 'bid_prc1']
        ask = df.loc[idx, 'ask_prc1']
        settle_prc = df.loc[idx, 'last_prc']
        pre_settle_prc = df.loc[idx - 1, 'last_prc'] if idx > 0 else None

        if vol:
            if bid and ask:
                df.loc[idx, 'middle'] = settle_prc if bid <= settle_prc <= ask else (bid + ask) / 2
            elif bid:
                df.loc[idx, 'middle'] = max(settle_prc, bid)
            elif ask:
                df.loc[idx, 'middle'] = min(settle_prc, ask)
            else:
                df.loc[idx, 'middle'] = settle_prc
        else:
            if bid and ask:
                df.loc[idx, 'middle'] = (bid + ask) / 2
            elif bid:
                df.loc[idx, 'middle'] = max(pre_settle_prc, bid) if pre_settle_prc is not None else bid
            elif ask:
                df.loc[idx, 'middle'] = min(pre_settle_prc, ask) if pre_settle_prc is not None else ask
            else:
                df.loc[idx, 'middle'] = pre_settle_prc

    # df.loc[:start_index + 1, 'acc_weight_price'] = 0  # 累计权重价格，vwap
    # cumulative_prc_volume = 0
    # cumulative_volume = 0
    #
    # for idx in range(start_index, len(df)):
    #     prc_volume = df.loc[idx, 'last_prc'] * df.loc[idx, 't_volume']
    #     volume = df.loc[idx, 't_volume']
    #     cumulative_prc_volume += prc_volume
    #     cumulative_volume += volume
    #     df.loc[idx, 'acc_weight_price'] = cumulative_prc_volume / cumulative_volume if cumulative_volume != 0 else \
    #     df.loc[idx, 'last_prc']
    # df['acc_weight_price'] = df['acc_weight_price'].round(2)

    df['stop'] = 0
    for idx, row in df.iterrows():
        if row['time'] >= 90000000:
            if ((row['last_prc'] == row['high_limited'] and row['ask_vol1'] == 0) or
                    (row['last_prc'] == row['low_limited'] and row['bid_vol1'] == 0)):
                df.loc[idx, 'stop'] = 1

    df = df.loc[(((df['time'] > 90000000) & (df['time'] <= 113000000)) |
                 ((df['time'] >= 130000000) & (df['time'] <= 150000000)))].copy()
    df.reset_index(drop=True, inplace=True)

    return df


def pre_process_quote_night(df):
    """
    quote   21:00 -
    :param df:
    :return:
    """

    def convert_int_to_time(time_int):
        time_str = str(int(time_int)).zfill(9)
        formatted_time_str = f'{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}.{time_str[6:]}'
        try:
            if len(formatted_time_str.split('.')) >= 3:
                formatted_time_str = formatted_time_str.split('.')[0] + '.' + formatted_time_str.split('.')[1]
            datetime_object = datetime.strptime(formatted_time_str, '%H:%M:%S.%f')
        except ValueError:
            print(f"{formatted_time_str} error")
            return None
        return datetime_object

    def convert_time_to_int(time_obj):
        time_str = time_obj.strftime('%H%M%S%f')[:-3]
        return int(time_str)

    def find_end_time(time_obj):
        end_time_list = [130000000, 143000000]  # 1:00, 2:30, 23:00
        closest_time = end_time_list[0]
        min_diff = abs(end_time_list[0] - time_obj)

        for time in end_time_list:
            diff = abs(time - time_obj)
            if diff < min_diff:
                min_diff = diff
                closest_time = time

        return closest_time

    df = df[df['night_trading'] == 1]  # 夜盘
    if df.empty:
        return pd.DataFrame()

    columns_to_keep = ['symbol', 'trading_date', 'date', 'time', 'recv_time', 'last_prc', 'open_interest', 'volume',
                       'turnover',
                       'ask_prc1', 'bid_prc1', 'ask_vol1', 'bid_vol1', 'acc_high', 'acc_low',
                       'high_limited', 'low_limited', 'prev_close', 'acc_open', 'night_trading']
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
    df.drop(columns=columns_to_drop, inplace=True)  # 去掉多余列
    df.sort_values(by='time', inplace=True)  # 按时间排序
    if 'recv_time' not in df.columns:
        df['recv_time'] = np.nan  # 加上receive time列

    if df['date'].nunique() > 1:
        mask1 = (df['time'] >= 205500000) & (df['time'] <= 240000000)
        mask2 = (df['time'] >= 0) & (df['time'] < 30000000)
        df.loc[mask1, 'time'] = df.loc[mask1, 'time'] - 120000000
        df.loc[mask2, 'time'] = df.loc[mask2, 'time'] + 120000000
        close_time = find_end_time(df['time'].max())

    else:
        mask1 = (df['time'] >= 205500000) & (df['time'] <= 230500000)
        df.loc[mask1, 'time'] = df.loc[mask1, 'time'] - 120000000
        close_time = 110000000

    if 90003000 not in df['time'].values:
        new_row = pd.DataFrame({
            'symbol': np.nan,
            'trading_date': np.nan,
            'date': np.nan,
            'time': [90003000],
            'recv_time': np.nan,
            'last_prc': np.nan,
            'open_interest': np.nan,
            'volume': np.nan,
            'turnover': np.nan,
            'ask_prc1': np.nan,
            'bid_prc1': np.nan,
            'ask_vol1': np.nan,
            'bid_vol1': np.nan,
            'acc_high': np.nan,
            'acc_low': np.nan,
            'high_limited': np.nan,
            'low_limited': np.nan,
            'prev_close': np.nan,
            'acc_open': np.nan,
            'night_trading': 1
        })
        df = pd.concat([df, new_row])
        df.sort_values(by='time', inplace=True)
        df.reset_index(drop=True, inplace=True)

    max_time_row = df[(df['time'] < 90003000)]['time'].idxmax()
    opening_call_row = df[df['time'] == 90003000]['time'].idxmax()
    max_time_values = df.loc[max_time_row, :]

    for col in df.columns:
        df.at[opening_call_row, col] = max_time_values[col]
    df.at[opening_call_row, 'time'] = 90003000

    if close_time not in df['time'].values:
        new_row = pd.DataFrame({
            'symbol': np.nan,
            'trading_date': np.nan,
            'date': np.nan,
            'time': [close_time],
            'recv_time': np.nan,
            'last_prc': np.nan,
            'open_interest': np.nan,
            'volume': np.nan,
            'turnover': np.nan,
            'ask_prc1': np.nan,
            'bid_prc1': np.nan,
            'ask_vol1': np.nan,
            'bid_vol1': np.nan,
            'acc_high': np.nan,
            'acc_low': np.nan,
            'high_limited': np.nan,
            'low_limited': np.nan,
            'prev_close': np.nan,
            'acc_open': np.nan,
            'night_trading': 1
        })

        df = pd.concat([df, new_row])
        df.sort_values(by='time', inplace=True)
        df.reset_index(drop=True, inplace=True)

    max_time_row = df[df['time'] >= close_time]['time'].idxmax()  # 最大time值对应的行号
    closing_call_row = df[df['time'] == close_time]['time'].idxmax()  # 3点对应的最大行号
    max_time_values = df.loc[max_time_row, :]  # 最大time值对应的行

    for col in df.columns:
        df.at[closing_call_row, col] = max_time_values[col]  # 把最大时间对应的行的数据放到3点对应的行
    df.at[closing_call_row, 'time'] = close_time


    df['time'] = df['time'].apply(convert_int_to_time)  # 转换成datetime
    df.dropna(subset=['time'], inplace=True)  # 去掉time空值的行

    if not (df.recv_time.isnull().all() or df.recv_time.isna().all()):
        df = df.dropna(subset=['recv_time'])  # 去掉receive_time空值的行
        idx_to_keep = df.groupby('time')['recv_time'].idxmin()  # receive时间并非全空值时，每个时间按receive的最小时间筛选
        df = df.loc[idx_to_keep].copy()
    else:
        idx_to_keep = df.groupby('time')['time'].idxmin()  # receive时间全空值时，每个时间按最小时间筛选
        df = df.loc[idx_to_keep].copy()


    df_resampled = df.set_index('time').resample('3S').asfreq()  # 时间设成索引，重采样，3s一个行情
    df_resampled.reset_index(inplace=True)
    df_resampled['time'] = df_resampled['time'].apply(convert_time_to_int)
    # df_resampled.set_index('time', inplace=True)
    df['time'] = df['time'].apply(convert_time_to_int)
    # df.set_index('time',inplace=True)
    pd.set_option('display.max_columns', None)  # 最大显示列数为无限
    df = pd.merge(df, df_resampled, how='outer')  # 按index合并，主要目的就是把每一个time对应一个重采样的时间
    df = df.sort_values('time')
    df.reset_index(drop=True, inplace=True)

    df['fill'] = df.apply(lambda row: 1 if row.isnull().any() else 0, axis=1)  # 对存在空值的行的fill列填1，否则0
    df['Time'] = df.apply(lambda x: x['time'] if x[['last_prc', 'volume']].notna().all() else np.nan,
                          axis=1)  # last_prc和volume不全为空时保留，否则nan

    last_non_na_rectime = None  # 上一个不是空值的receive time，用于填充下一个
    last_non_na_time = None  # 上一个不是空值的time，用于填充下一个

    for idx, row in df.iterrows():
        if pd.notna(row['recv_time']):
            last_non_na_rectime = row['recv_time']
        else:
            df.at[idx, 'recv_time'] = last_non_na_rectime

        if pd.notna(row['Time']):
            last_non_na_time = row['Time']
        else:
            df.at[idx, 'Time'] = last_non_na_time

    df = df.sort_values('time')
    df.ffill(inplace=True)

    df['time'] = df['time'].apply(convert_int_to_time)
    df = df.set_index('time').resample('3S').asfreq()
    df.ffill(inplace=True)
    df.reset_index(inplace=True)
    df['time'] = df['time'].apply(convert_time_to_int)

    index_location1 = df[df['time'] >= 90000000].index.min()  # 取对应索引
    if pd.isna(index_location1):  # Check if index_location1 is NaN
        pd.DataFrame()
    # try:
    #     row_number1 = df.index.get_loc(index_location1)  # 取行号，其实多此一举了，因为已经reset_index了
    # except KeyError:
    #     print(f"Index {index_location1} not found.")
    #     pd.DataFrame()

    index_location2 = df[df['time'] <= close_time].index.max()
    # row_number2 = df.index.get_loc(index_location2)
    df = df.loc[index_location1: index_location2].reset_index(drop=True)  # 取9-15区间内数据

    df = df.sort_values('time')
    start_index = df[df['time'] >= 90000000].index[0]
    df['t_volume'] = 0
    for idx in range(start_index, len(df)):
        if idx == 0:
            df.loc[idx, 't_volume'] = 0
        else:
            df.loc[idx, 't_volume'] = df.loc[idx, 'volume'] - df.loc[idx - 1, 'volume']  # 时刻的volume（原始数据为累计volume）
    df['t_volume'] = df['t_volume'].fillna(0)

    df = df.sort_values('time')
    df['middle'] = None  # 中间价
    df.loc[:start_index + 1, 'middle'] = 0
    for idx in range(start_index, len(df)):
        vol = df.loc[idx, 't_volume']
        bid = df.loc[idx, 'bid_prc1']
        ask = df.loc[idx, 'ask_prc1']
        settle_prc = df.loc[idx, 'last_prc']
        pre_settle_prc = df.loc[idx - 1, 'last_prc'] if idx > 0 else None

        if vol:
            if bid and ask:
                df.loc[idx, 'middle'] = settle_prc if bid <= settle_prc <= ask else (bid + ask) / 2
            elif bid:
                df.loc[idx, 'middle'] = max(settle_prc, bid)
            elif ask:
                df.loc[idx, 'middle'] = min(settle_prc, ask)
            else:
                df.loc[idx, 'middle'] = settle_prc
        else:
            if bid and ask:
                df.loc[idx, 'middle'] = (bid + ask) / 2
            elif bid:
                df.loc[idx, 'middle'] = max(pre_settle_prc, bid) if pre_settle_prc is not None else bid
            elif ask:
                df.loc[idx, 'middle'] = min(pre_settle_prc, ask) if pre_settle_prc is not None else ask
            else:
                df.loc[idx, 'middle'] = pre_settle_prc

    # df.loc[:start_index + 1, 'acc_weight_price'] = 0  # 累计权重价格，vwap
    # cumulative_prc_volume = 0
    # cumulative_volume = 0
    #
    # for idx in range(start_index, len(df)):
    #     prc_volume = df.loc[idx, 'last_prc'] * df.loc[idx, 't_volume']
    #     volume = df.loc[idx, 't_volume']
    #     cumulative_prc_volume += prc_volume
    #     cumulative_volume += volume
    #     df.loc[idx, 'acc_weight_price'] = cumulative_prc_volume / cumulative_volume if cumulative_volume != 0 else \
    #     df.loc[idx, 'last_prc']
    # df['acc_weight_price'] = df['acc_weight_price'].round(2)

    df['stop'] = 0
    for idx, row in df.iterrows():
        if row['time'] >= 90000000:
            if ((row['last_prc'] == row['high_limited'] and row['ask_vol1'] == 0) or
                    (row['last_prc'] == row['low_limited'] and row['bid_vol1'] == 0)):
                df.loc[idx, 'stop'] = 1


    mask1 = (df['time'] > 85500000) & (df['time'] <= 120000000)
    df.loc[mask1, 'time'] = df.loc[mask1, 'time'] + 120000000
    mask2 = (df['time'] > 120000000) & (df['time'] < 150000000)
    if not df.loc[mask2].empty:
        df.loc[mask2, 'time'] = df.loc[mask2, 'time'] - 120000000

    df.reset_index(drop=True, inplace=True)

    return df