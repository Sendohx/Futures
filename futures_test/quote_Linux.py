import os
import sys
import time
import pandas as pd
import warnings

from get_main_contract_wind import get_main_contract
from futures_3s import pre_process_quote_day, pre_process_quote_night
from Common.logger import get_logger
from ApiClient import ApiClient

warnings.filterwarnings('ignore')
# sys.path.append(r".\py311")   # really important


if __name__ == '__main__':
    G_LOGGER = get_logger(name="ApiClientLog", debug=False)

    api = ApiClient("zlt_read", "zlt_read", "config/api.json", G_LOGGER)

    if not api.init():
        print("初始化失败")
        sys.exit(0)

    if True:
        # futures_code = 'cu'
        # dict = {
        #         'symbol': ['CU2302.SHF', 'CU2303.SHF','CU2304.SHF','CU2305.SHF','CU2306.SHF','CU2307.SHF',
        #                  'CU2308.SHF','CU2309.SHF','CU2310.SHF','CU2311.SHF','CU2312.SHF','CU2401.SHF'],
        #         'start': ['2023-01-01', '2023-01-09', '2023-02-23', '2023-03-21', '2023-04-25', '2023-05-22',
        #                   '2023-06-28', '2023-07-26', '2023-08-24', '2023-09-25', '2023-10-23', '2023-11-22'],
        #         'end': ['2023-01-06', '2023-02-22', '2023-03-20', '2023-04-24','2023-05-19', '2023-06-27',
        #                 '2023-07-25', '2023-08-23', '2023-09-22', '2023-10-20', '2023-11-21', '2023-12-20'],
        #         }
        # dict = {
        #         'symbol': ['CU2304.SHF'],
        #         'start': ['2023-02-23'],
        #         'end': ['2023-03-20']
        # }
        # contract_df = pd.DataFrame(dict)   # 主力合约的df可在wind查询
        # row = contract_df.iloc[0]
        # df = api.file_read([row['symbol']], row['start'], row['end'])[0]
        save_path = '/nas92/data/future'
        symbol_list = ['CU', 'CF', 'LC', 'SC', 'IM', 'T', 'I']   #剩CF
        start_date = '20230101'
        end_date = '20231231'
        result = pd.DataFrame(columns=['symbol', 'contract', 'start', 'end'])

        for symbol in symbol_list:
            mc = get_main_contract(symbol, start_date, end_date)
            temp = mc.run()
            result = pd.concat([result, temp])

        mark1 = result['start'] <= start_date
        mark2 = result['end'] >= end_date
        result.loc[mark1, 'start'] = start_date
        result.loc[mark2, 'end'] = end_date

        result.sort_values(by=['symbol', 'start'], inplace=True)
        result.reset_index(drop=True, inplace=True)

        result.loc[result['symbol'] == 'CF', 'contract'] = (result.loc[result['symbol'] == 'CF', 'contract'].str.
                                                            replace(r'(\D+)(\d{3})(\..+)', r'\g<1>2\g<2>\g<3>', regex=True))
        result['start'] = pd.to_datetime(result['start'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        result['end'] = pd.to_datetime(result['end'], format='%Y%m%d').dt.strftime('%Y-%m-%d')




        for index, row in result.iterrows():
            print('\n' + row['contract'] + ' start')
            df = api.file_read([row['contract']], row['start'], row['end'])[0]
            # df['date'] = pd.to_datetime(df['datetime'])
            # df['date'] = df['date'].dt.date
            try:
                df['night_trading'] = (((df['time'] >= 205500000) & (df['time'] <= 240000000)) | (
                        (df['time'] >= 0) & (df['time'] <= 23500000))).astype(int)
            except Exception as e:
                print(e)

            start_date = int(row['start'].replace('-', ''))
            end_date = int(row['end'].replace('-', ''))
            current_date = start_date

            while current_date <= end_date:
                s_time = time.time()
                temp = df[df['trading_date'] == current_date]

                if temp.empty:
                    print(f'no data for {current_date}' + ' ' + row['contract'])
                else:
                    if not os.path.exists(save_path + f'/{current_date}'):
                        os.mkdir(save_path + f'/{current_date}')

                    if os.path.exists(save_path + f'/{current_date}/3s_quote_' + row['contract'] + '.parquet'):
                        print(f'{current_date} -- 3s_quote_'  + row['contract'] + ' exists, pass')
                        current_date += 1
                        continue

                    if os.path.exists(save_path + f'/{current_date}/3s_quote(N)_' + row['contract'] + '.parquet'):
                        print(f'{current_date} -- 3s_quote(N)_' + row['contract'] + ' exists, pass')
                        current_date += 1
                        continue

                    temp_day = pre_process_quote_day(temp)

                    if temp['night_trading'].nunique() > 1:
                        temp_night = pre_process_quote_night(temp)
                        result = pd.concat([temp_day, temp_night])
                        result.to_parquet(save_path + f'/{current_date}/3s_quote(N)_' + row['contract'] + '.parquet')
                        e_time = time.time()
                        print(f'{current_date} -- 3s_quote(N)_' + row['contract'] + f' finish, running time: {e_time - s_time: .6f} seconds')
                    else:
                        print(f'no night trading on {current_date}')
                        temp_day.to_parquet(save_path + f'/{current_date}/3s_quote_' + row['contract'] + '.parquet')
                        e_time = time.time()
                        print(f'{current_date} -- 3s_quote_' + row['contract'] + f' finish, running time: {e_time - s_time: .6f} seconds')

                current_date += 1


            contract = row['contract']
            print(f'{contract} finish')


        print("finish.")

