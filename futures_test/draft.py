import pandas as pd
pd.set_option('display.max_columns', None)

a = pd.read_parquet('/nas92/data/future/20230103/3s_quote_CU2302.SHF.parquet')
print(a)