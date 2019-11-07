import torch
import pandas as pd
from torch.utils import data
from bisect import bisect_left


class Dataset(data.Dataset):
    def __init__(self, start_date: str, end_date: str, product: str, data_type: str,
                 series_length: int, x_fields: list, y_field: str, 
                 sample_interval: int, cache_limit: int, use_cuda: bool=True):
        """
        Initialization
        :param start_date: start date of dataset in YYYYMMDD format
        :param end_date: end date of dataset in YYYYMMDD format
        :param product: product / directory name
        :param data_type: train, validate or test
        :param series_length: number of adjacent ticks to pass
        :param x_fields: columns to use as X
        :param y_field: columns to use as Y
        :param sample_interval: sample every x seconds
        :param cache_limit: number of files that are kept in memory
        :param use_cuda: flag to use GPU or not
        """
        self.start_date = start_date
        self.end_date = end_date
        self.product = product  # i.e. IH
        self.data_type = data_type
        self.series_length = series_length
        self.x_fields = x_fields
        self.y_field = y_field
        self.sample_interval = sample_interval
        
        self.tick_info_cols = ['datetime', 'bid1', 'ask1']

        self.index_table = self._get_index_table()

        # cache variables
        self.cache_limit = cache_limit
        self.cache_dates = []  # queue
        self.cache_dfs = {}
        
        self.use_cuda = use_cuda

    def __len__(self):
        """Denotes the total number of samples"""
        return self.index_table['acc_rows'].iloc[-1]

    def __getitem__(self, index):
        """Generates one sample of data with series_length ticks"""
        index_i = self._find_index(self.index_table['acc_rows'], index)
        date = self.index_table['date'].iloc[index_i]
        fn = self.index_table['file_name'].iloc[index_i]
        row_num = self.index_table['rows'].iloc[index_i]
        acc_rows = self.index_table['acc_rows'].iloc[index_i]

        # check if data in cache
        if date in self.cache_dfs:
            df = self.cache_dfs[date]
        else:
            if len(self.cache_dates) >= self.cache_limit:
                remove_date = self.cache_dates.pop(0)
                self.cache_dfs.pop(remove_date)
            df = pd.read_csv(f'{self.product}/{fn}', usecols=self.x_fields+[self.y_field]+self.tick_info_cols)
            self.cache_dfs[date] = df
            self.cache_dates.append(date)

        # Select intraday sample
        
        intra_index = row_num - (acc_rows - index) + 1  # skip first tick
#         print(date, fn, row_num, acc_rows, index, intra_index)
        
#         try:
        sample_ticks = int(self.sample_interval / 0.5)
        total_ticks = self.series_length * sample_ticks
        X_df = df[self.x_fields].iloc[intra_index: intra_index + total_ticks: sample_ticks]
        X_raw = X_df.values.reshape(
            (self.series_length, 1, len(self.x_fields))
        ).astype('float64')
        X = torch.from_numpy(X_raw)
        y = torch.tensor(float(df[self.y_field].iloc[intra_index + total_ticks - 1]))

#         last_sample_tick_info = df[self.tick_info_cols].iloc[intra_index + total_ticks - sample_ticks].values  # for pnl calc
        
        return X, y

    def _get_index_table(self):
        index_table = pd.read_csv(
            f'{self.product}/{self.product}_{self.data_type}_indexes_{self.series_length}_{self.sample_interval}.csv'
        )
        index_table['date'] = index_table['date'].astype(str)

        start_i = self._find_index(index_table['date'], self.start_date)
        end_i = self._find_index(index_table['date'], self.end_date, roundup=True)
        index_table = index_table.iloc[start_i: end_i+1]
        index_table['acc_rows'] = index_table['rows'].cumsum()

        return index_table

    @staticmethod
    def _find_index(lst, x, roundup=False):
        """
        Locate the leftmost value
        :param roundup: if true, find the leftmost value
        :return: index
        """
#         print(lst, x)
        i = bisect_left(lst, x)
        if i == len(lst):
            return i - 1
        if roundup and i != len(lst) - 1:
            return i + 1
        return i
