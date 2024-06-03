import sys
import datetime
import pandas as pd
from typing import List

from logbook import Logger
from Common.logger import get_logger
from DataApi import CrsApi


class ApiClient(CrsApi):
    """
    api 客户端头文件
    """

    def __init__(self, user: str, passwd: str, config_path: str, log_inst: Logger, thread_num: int = 16):
        """
        user:登录用户
        passwd:登录密码
        config_path:配置文件路径
        log_inst:日志实例
        thread_num:客户端解析数据线程数量
        """
        super().__init__(user, passwd, config_path, log_inst, thread_num)

    def init(self) -> bool:
        """
        DataApi客户端初始化 调用后可以使用
        """
        return self._init()

    def stop(self):
        """
        实例停止 资源回收
        """
        self._stop()

    def query_history(self, entries: List[str], entry_type: str, start_time: str, end_time: str,
                      fields: List[str] = None):
        """
        查询历史数据
        entries:需要查询的条目名，可以是标的名列表，也可以是扩展的因子名列表, eg:["000001.SZ","000002.SZ"]、["MOM_7", "MOM_14"]
        entry_type:查询数据的类型，支持：quote、trade、order、minbar、daybar...
        start_time:开始时间两种格式
                        "yyyy-mm-dd HH:MM:SS", eg:"2023-9-15 09:30:00"
                        "yyyy-mm-dd", eg:"2023-09-15"
        end_time:结束时间两种格式
                        "yyyy-mm-dd HH:MM:SS", eg:"2023-9-15 14:30:00"
                        "yyyy-mm-dd", eg:"2023-09-16"
        fields:查询的字段名称, eg:["open", "high", "low", "close"]
        """
        return self._query_history(entries=entries, entry_type=entry_type, start_time=start_time, end_time=end_time,
                                   fields=fields)

    def query_from_sql(self, query_sql: str, tbl_name: str):
        """
        查询wind数据库
        query_sql:查询sql语句
        tbl_name:待查询的表名
        """
        return self._query_from_sql(query_sql=query_sql, tbl_name=tbl_name)


    def query_and_output_result(self, entry_list: List[str], entry_type: str, test_start_time, test_end_time):
        entry_type_result = self.query_history(entry_list, entry_type, test_start_time, test_end_time)

        total_len = 0
        for result in entry_type_result:
            if result is None:
                continue
            else:
                length = len(result)
                print("{} len:{}".format(entry_type, length))
                total_len += length
        print("{}总长度:{}".format(entry_type, total_len))

        return entry_type_result


    def file_read(self, symbol_list, test_start_time, test_end_time):
        start = datetime.datetime.now()
        """
        查询接口测试用例
        """
        # 查询快照
        result = self.query_and_output_result(symbol_list, "quote", test_start_time, test_end_time)

        # 查询逐笔成交
        # query_and_output_result(symbol_list, "trade")

        # 查询逐笔委托
        # query_and_output_result(symbol_list, "order")

        # 查询分钟bar
        # self.query_and_output_result(symbol_list, "minbar", test_start_time, test_end_time)

        end = datetime.datetime.now()
        print("耗时:{}s".format((end - start).total_seconds()))

        return result

    def sql_read(self, symbol_list, test_start_time, test_end_time):
        # 查询万得day bar
        self.query_and_output_result(symbol_list, "daybar", test_start_time, test_end_time)

        # 查询万得某张表数据
        calendar_table = "ASHARECALENDAR"
        query_calendar = "select TRADE_DAYS from {} where S_INFO_EXCHMARKET = 'SSE' order by TRADE_DAYS".format(
            calendar_table)
        calendar = self.query_from_sql(query_calendar, calendar_table)

        for result in calendar:
            if result is None:
                continue
            else:
                print("{}".format(result.head()))


def get_symbol_list():
    read_symbols = []
    read_file = False
    if read_file:
        tick_df = pd.read_csv("./trade_tick.csv")
        read_symbols = list(tick_df['symbol'])
    else:
        read_symbols = []
    return read_symbols


# if __name__ == '__main__':
#     G_LOGGER = get_logger(name="ApiClientLog", debug=False)
#
#     api = ApiClient("zlt_read", "zlt_read", "config/api.json", G_LOGGER)
#
#     if not api.init():
#         print("初始化失败")
#         sys.exit(0)
#
#     if True:
#         test_start_time = "2023-01-04 9:00:00"
#         test_end_time = "2023-01-04 15:30:00"
#     else:
#         test_start_time = "2023-01-04"
#         test_end_time = "2023-01-05"
#
#     # 获取测试的标的
#     symbol_list = get_symbol_list()
#
#     # 测试文件数据库
#     file_read()
#
#     # 测试Mysql数据库
#     # sql_read()
#
#     api.stop()
