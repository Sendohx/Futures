# -*- coding:UTF-8 -*-

"""
文件说明：
"""
import os
import sys
import glob
import warnings
import concurrent.futures
import pandas as pd
from datetime import datetime, timedelta


warnings.simplefilter('ignore')

class base_data():
    def __init__(self):
        self.mongo_username = "zlt01"
        self.mongo_password = "zlt_ujYH"
        self.mongo_host = "mongodb://zlt01:zlt_ujYH@192.168.9.189:15009/data"

        self.database85_username = 'chuangXin'
        self.database85_password = 'Xini.100'
        self.database85_host = '192.168.9.85'
        self.database85_basename = 'option_new'

        self.wind_username = 'quantchina'
        self.wind_password = 'zMxq7VNYJljTFIQ8'
        self.wind_host = '192.168.7.93'
        self.wind_port = 3306

        self.data_path = "//192.168.7.92/data/"

    def get_tick_data(self, symbol, start_date, end_date, symbol_type="future"):
        """
        从datahouse中获取tick
        :param symbol: 合约代码，str,format like IF2109.CFE
        :param date : str,format like '20210907'
         param symbol_type : str, must be 'future' or 'stock'
        :return: dataframe
        """

        start = datetime.strptime(start_date, "%Y%m%d").date()
        end = datetime.strptime(end_date, "%Y%m%d").date()
        date_range = (end - start).days + 1
        df = pd.DataFrame()

        try:
            code = (symbol.split('.')[0]).lower()
            exchange = (symbol.split('.')[1]).lower()
        except Exception as e:
            print(e)
            return df

        if symbol_type == "future":
            for i in range(date_range):
                current_date = start + timedelta(days=i)
                current_date_str = current_date.strftime('%Y%m%d')
                tick_dir = self.data_path + "/tick/" + symbol_type + "/" + current_date_str + "/quote/"
                file_name = exchange + "_" + code + "_" + current_date_str + "_quote.parquet"
                if os.path.exists(tick_dir + file_name):
                    df = pd.concat([df, pd.read_parquet(tick_dir + file_name)], axis=0)
                    # print("tick data on ", current_date_str)
                else:
                    print("no data on ", current_date_str)
            try:
                df['night_trading'] = (((df['time'] >= 210000000) & (df['time'] < 240000000)) | (
                            (df['time'] > 0) & (df['time'] <= 1000000))).astype(int)
            except Exception as e:
                print(e)

        #if symbol_type == "stock":

            return df


    def get_minbar_data(self, symbol, start_date, end_date, symbol_type="future"):
        """
        从datahouse中获取分钟bar，
        :param symbol: str,format like IF2109.CFE or 600001.SH
        :param date : str,format like '20210907'
        :param symbol_type : str, must be 'future' or 'stock'
        :return: dataframe
        """
        start = datetime.strptime(start_date, "%Y%m%d").date()
        end = datetime.strptime(end_date, "%Y%m%d").date()
        date_range = (end - start).days + 1
        df = pd.DataFrame()

        try:
            code = (symbol.split('.')[0]).lower()
            exchange = (symbol.split('.')[1]).lower()
        except Exception as e:
            print(e)
            return df
        if symbol_type == "future" or symbol_type == "stock":
            for i in range(date_range):
                current_date = start_date + timedelta(days = i)
                current_date_str = current_date.strftime('%Y%m%d')
                min_bar_dir = self.data_path + "/minbar/" + symbol_type + "/" + current_date_str + "/1min/"
                file_name = exchange + "_" + code + "_" + current_date_str + "_1min.parquet"
                if os.path.exists(min_bar_dir + file_name):
                    df = pd.concat([df, pd.read_parquet(min_bar_dir + file_name)], axis=0)
                else:
                    print("min bar path error:", min_bar_dir + file_name)
        return df


    def get_csv_summary(self, head, date, symbol_type="future"):
        df = pd.DataFrame()
        summary_dir = self.data_path + "/tick//" + symbol_type + "//" + date + "/"
        file_name = head + "_quote_summary.csv"
        if os.path.exists(summary_dir + file_name):
            df = pd.read_csv(summary_dir + file_name)
        else:
            print("tick data path error:", summary_dir + file_name)
        return df


class main_contract():
    def __init__(self, futures_code, start_date, end_date, symbol_type="future"):
        """
        参数:
        future_data: str 品种代码
        start_date: str 起始日期（格式：YYYY-MM-DD）
        end_date: str 截止日期（格式：YYYY-MM-DD）

        """
        self.data_path = "//192.168.7.92/data/"
        self.futures_code = futures_code
        self.start_date = start_date
        self.end_date = end_date
        self.symbol_type = symbol_type
        # super().__init__("zlt_read", "zlt_read", "config/api.json", get_logger(name="ApiClientLog", debug=False))

    def process_data(self, current_date, tick_dir, futures_code):
        current_date_str = current_date.strftime('%Y%m%d')
        # tick_dir = self.data_path + "/tick/" + self.symbol_type + "/" + current_date_str + "/quote/"
        pattern = os.path.join(tick_dir, f"???_{futures_code}????_{current_date_str}_quote.parquet")
        files_list = glob.glob(pattern)
        data = []

        for file in files_list:
            try:
                temp = pd.read_parquet(file)
                data.append(temp.loc[temp['open_interest'].idxmax()])
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        if data:
            df = pd.DataFrame(data)
            print(current_date_str, " finish.")
            df['date'] = current_date_str  # 夜盘归类到第二天
            return df.sort_values(by=['open_interest'], ascending=False).head(1)
        else:
            return pd.DataFrame()

    def get_main_contract_tick(self):
        """
        查询期货的主力合约

        返回:
        DataFrame 包含日期、合约代码和换月标记
        """
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        date_range = (end - start).days + 1

        if self.symbol_type == "future":
            date_list = [start + timedelta(days=i) for i in range(date_range)]
            tick_dirs = [os.path.join(self.data_path, "tick", self.symbol_type, date.strftime('%Y%m%d'), "quote") for
                         date in date_list]
            futures_code = self.futures_code
            main_contract_list = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as executor:
                future_to_date = {executor.submit(self.process_data, date, tick_dir, futures_code): date for
                                  date, tick_dir in zip(date_list, tick_dirs)}
                for future in concurrent.futures.as_completed(future_to_date):
                    try:
                        result = future.result()
                        if not result.empty:
                            main_contract_list.append(result)
                    except Exception as e:
                        print(f"Error processing data for date {future_to_date[future]}: {e}")

            if main_contract_list:
                main_contract = pd.concat(main_contract_list, ignore_index=True)
                main_contract['date'] = pd.to_datetime(main_contract['date'], format='%Y%m%d')
                main_contract.sort_values(by='date', inplace=True)

                main_contract['code'] = main_contract['symbol'].apply(lambda x: x.split('.')[0])
                main_contract['month'] = main_contract['code'].apply(lambda x: x[-4:])
                main_contract['shifted_month'] = main_contract['month'].shift(1)
                main_contract['switch_month_flag'] = main_contract['month'] != main_contract['shifted_month']
                main_contract['switch_month_flag'] = main_contract['switch_month_flag'].astype(int)
                main_contract['switch_month_flag'].iloc[0] = 0

                return main_contract[['date', 'symbol', 'switch_month_flag']].reset_index(drop=True)
            else:
                print("No data found for the given date range.")
                return pd.DataFrame()

        # if self.symbol_type == "stock":
