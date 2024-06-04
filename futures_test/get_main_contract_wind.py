import pandas as pd

from futures_test.connect_wind import ConnectDatabase

# 获取历史主力对应月合约
class get_main_contract(ConnectDatabase):

    def __init__(self, future_symbol, start_date: str, end_date: str):
        self.future_symbol = future_symbol  #例 铜CU(大写)
        self.start_date = start_date
        self.end_date = end_date
        self.sql = f'''
                    SELECT S_INFO_WINDCODE, FS_MAPPING_WINDCODE, STARTDATE, ENDDATE
                    FROM CFUTURESCONTRACTMAPPING
                    WHERE ((STARTDATE BETWEEN {self.start_date} AND {self.end_date}) 
                    OR (ENDDATE BETWEEN '{self.start_date}' AND '{self.end_date}'))
                    AND S_INFO_WINDCODE like '{future_symbol}.%'
                    '''

        super().__init__(self.sql)
        self.df = super().get_data()
        # self.df = self.df[['FS_MAPPING_WINDCODE', 'STARTDATE', 'ENDDATE']]
        self.df = self.df.rename(columns={'S_INFO_WINDCODE': 'symbol',
                                          'FS_MAPPING_WINDCODE': 'contract',
                                          'STARTDATE': 'start',
                                          'ENDDATE': 'end'})
        self.df['symbol'] = self.df['symbol'].str.split('.').str[0]
        self.df.sort_values(by='start', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def  run(self):
        return self.df



if __name__ == '__main__':
    # 铜CU， 棉花CF， 碳酸锂LC， 中质含硫原油SC， 中证1000指数IM， 10年期国债期货T，铁矿石I
    symbol_list = ['CU', 'LC', 'SC', 'IM', 'T', 'I']  #, 'CF'
    start_date = '20230101'
    end_date = '20231231'
    result = pd.DataFrame(columns=['symbol', 'contract', 'start', 'end'])

    for symbol in symbol_list:
        mc = get_main_contract(symbol, start_date, end_date)
        temp = mc.run()
        result = pd.concat([result, temp])

    # result['start'] = result['start'].astype(int)
    # result['end'] = result['end'].astype(int)

    mark1 = result['start'] <= start_date
    mark2 = result['end'] >= end_date
    result.loc[mark1, 'start'] = start_date
    result.loc[mark2, 'end'] = end_date

    result.sort_values(by=['symbol', 'start'], inplace=True)
    result.reset_index(drop=True, inplace=True)

    result.loc[result['symbol'] == 'CF', 'contract'] = (result.loc[result['symbol'] == 'CF', 'contract'].
                                                        str.replace(r'(\D+)(\d{3})(\..+)', r'\g<1>2\g<2>\g<3>', regex=True))
    result['start'] = pd.to_datetime(result['start'], format= '%Y%m%d').dt.strftime('%Y-%m-%d')
    result['end'] = pd.to_datetime(result['end'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

    print(result)