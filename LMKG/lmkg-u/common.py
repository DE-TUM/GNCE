"""
    Title: Deep Unsupervised Cardinality Estimation Source Code
    Author:  Amog Kamsetty, Chenggang Wu, Eric Liang, Zongheng Yang
    Date: 2020
    Availability: https://github.com/naru-project/naru

    Source Code used as is or modified from the above mentioned source
"""

import copy
import time

import numpy as np
import pandas as pd
from compressor import Compressor

import torch
from torch.utils import data

# Na/NaN/NaT Semantics
#
# Some input columns may naturally contain missing values.  These are handled
# by the corresponding numpy/pandas semantics.
#
# Specifically, for any value (e.g., float, int, or np.nan) v:
#
#   np.nan <op> v == False.
#
# The above evaluation is consistent with SQL semantics.


class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('Attr1').Fill(data, infer_dist=True)

    The passed-in 'data' is copied by reference.
    """

    def __init__(self, name, distribution_size=None, pg_name=None):
        self.name = name

        # Data related fields.
        self.data = None
        self.all_distinct_values = None
        self.distribution_size = distribution_size

        # pg_name is the name of the corresponding column in Postgres.  This is
        # put here since, e.g., PG disallows whitespaces in names.
        self.pg_name = pg_name if pg_name else name

    def Name(self):
        """Name of this column."""
        return self.name

    def DistributionSize(self):
        """This column will take on discrete values in [0, N).

        Used to dictionary-encode values to this discretized range.
        """
        return self.distribution_size

    def ValToBin(self, val):
        if isinstance(self.all_distinct_values, list):
            return self.all_distinct_values.index(val)
        inds = np.where(self.all_distinct_values == val)
        assert len(inds[0]) > 0, val

        return inds[0][0]

    def SetDistribution(self, distinct_values):
        """This is all the values this column will ever see."""
        assert self.all_distinct_values is None
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(distinct_values)
        contains_nan = np.any(is_nan)
        dv_no_nan = distinct_values[~is_nan]
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(dv_no_nan))
        if contains_nan and np.issubdtype(distinct_values.dtype, np.datetime64):
            vs = np.insert(vs, 0, np.datetime64('NaT'))
        elif contains_nan:
            vs = np.insert(vs, 0, np.nan)
        if self.distribution_size is not None:
            assert len(vs) == self.distribution_size
        self.all_distinct_values = vs
        self.distribution_size = len(vs)
        return self

    def Fill(self, data_instance, infer_dist=False):
        assert self.data is None
        self.data = data_instance
        # If no distribution is currently specified, then infer distinct values
        # from data.
        if infer_dist:
            self.SetDistribution(self.data)
        return self

    def __repr__(self):
        return 'Column({}, distribution_size={})'.format(
            self.name, self.distribution_size)


class Table(object):
    """A collection of Columns."""

    def __init__(self, name, columns, pg_name=None):
        """Creates a Table.

        Args:
            name: Name of this table object.
            columns: List of Column instances to populate this table.
            pg_name: name of the corresponding table in Postgres.
        """
        self.name = name
        self.cardinality = self._validate_cardinality(columns)
        self.columns = columns

        self.val_to_bin_funcs = [c.ValToBin for c in columns]
        self.name_to_index = {c.Name(): i for i, c in enumerate(self.columns)}

        if pg_name:
            self.pg_name = pg_name
        else:
            self.pg_name = name

    def __repr__(self):
        return '{}({})'.format(self.name, self.columns)

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [len(c.data) for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def Name(self):
        """Name of this table."""
        return self.name

    def Columns(self):
        """Return the list of Columns under this table."""
        return self.columns

    def ColumnIndex(self, name):
        """Returns index of column with the specified name."""
        assert name in self.name_to_index
        return self.name_to_index[name]


class CsvTable(Table):
    """Wraps a CSV file or pd.DataFrame as a Table."""

    def __init__(self,
                 name,
                 filename_or_df,
                 cols,
                 type_casts={},
                 pg_name=None,
                 pg_cols=None,
                 do_compression=None,
                 num_submterms=2,
                 comp_threshold=1000,
                 **kwargs):
        """Accepts the same arguments as pd.read_csv().

        Args:
            filename_or_df: pass in str to reload; otherwise accepts a loaded
              pd.Dataframe.
            cols: list of column names to load; can be a subset of all columns.
            type_casts: optional, dict mapping column name to the desired numpy
              datatype.
            pg_name: optional str, a convenient field for specifying what name
              this table holds in a Postgres database.
            pg_name: optional list of str, a convenient field for specifying
              what names this table's columns hold in a Postgres database.
            do_compression: boolean, perform compression on the terms.
            num_submterms: integer, the number of subterms the term will be split into.
            comp_threshold: integer, threshold for the required number of unique values
            per term for qualification for compression.
            **kwargs: keyword arguments that will be pass to pd.read_csv().
        """
        self.name = name
        self.pg_name = pg_name

        if do_compression:
            # Compression. This means that the column will be split into 2 columns
            root_used_for_divison = num_submterms
            self.compressor_element = Compressor(root_used_for_divison)

        if isinstance(filename_or_df, str):
            self.data, cols = self._load(filename_or_df, cols, doCompression=do_compression, compression_threshold=comp_threshold, **kwargs)
        else:
            assert (isinstance(filename_or_df, pd.DataFrame))
            self.data = filename_or_df

        self.columns = self._build_columns(self.data, cols, type_casts, pg_cols)

        super(CsvTable, self).__init__(name, self.columns, pg_name)

    def call_divide_column(self, column_values, column_divider, original_col_index):
        return self.compressor_element.divide_column(column_values, column_divider, original_col_index)

    def compressData(self, original_df, cols, root, when_to_compress):
        '''
        Method for compressing the data. Every column that has more unique values then 'when_to_compress' is split
        into 'root' columns.

        :param original_df:
        :param cols:
        :param root:
        :param when_to_compress:
        :return:
        '''
        compressed_data = pd.DataFrame()
        boundries_per_column = dict()
        modified_columns = []
        for col in cols:
            # extract the maximal value for the column
            max_column_value = original_df[col].max()
            # if the value satisfies the requirement then calculate the divider and update the column names
            if max_column_value > when_to_compress:
                boundries_per_column[col] = int(max_column_value ** (1/root))
                for i in range(root):
                    modified_columns.append(col + '' + str(i+1))
            else:
                modified_columns.append(col)

        current_col_title = 0
        for i, col in enumerate(cols):
            data_column = original_df[col]

            if col in boundries_per_column:
                print('compressing column: %s' % col)
                # every column at the beginning will be split into 2 columns
                how_many_times_compressed = 2
                # for every column that has the potential to be split, perform the split
                try:
                    quotient_column, reminder_column = self.call_divide_column(data_column.values, boundries_per_column[col], i)
                except:
                    print(data_column.values)
                    print(boundries_per_column[col])
                    print(i)
                    exit(1)
                # list of all the reminders that we'll need at the end
                all_reminders = list()
                # add the first reminder which will actually represent the last column
                all_reminders.append(reminder_column)

                # if the number of current columns is different than the number of columns that we want to have perform the split
                while how_many_times_compressed < root:
                    quotient_column, reminder_column = self.call_divide_column(quotient_column,
                                                                               boundries_per_column[col], i)
                    # store the reminder
                    all_reminders.append(reminder_column)
                    # increase the number of columns
                    how_many_times_compressed += 1
                # part for creating the columns
                compressed_data[modified_columns[current_col_title]] = quotient_column
                current_col_title += 1
                # for the reminder columns, the last columns should actually go first
                for rem_enum, rem in enumerate(reversed(all_reminders)):
                    compressed_data[modified_columns[current_col_title]] = rem
                    if rem_enum + 1 < len(all_reminders):
                        current_col_title += 1
            else:
                # for the columns that should't be split, add them to the correct place
                compressed_data[modified_columns[current_col_title]] = data_column.values

            # go to the next column
            current_col_title += 1
        print('shape of compressed data:', end=' ')
        print(np.shape(compressed_data))

        return compressed_data, modified_columns



    def _load(self, filename, cols, doCompression=False, compression_threshold=1000, **kwargs):
        print('Loading csv...', end=' ')
        print()
        s = time.time()
        df = pd.read_csv(filename, usecols=cols, **kwargs)
        print('original data shape:', end=' ')
        print(np.shape(df))
        modified_cols = cols
        if doCompression:
            '''
                Create a compression where we split the column into two columns such that we get the root
                closest to the maximal number of the column. 
                Using that we divide every number in that column with the square root and we get 
                the multiplier and the quotient. 
            '''
            # represents the required number of unique values to qualify a column for compression
            min_num_unique_domain_values_column_to_qualify = compression_threshold
            df, modified_cols = self.compressData(df,cols, self.compressor_element.root, min_num_unique_domain_values_column_to_qualify)
        else:
            if cols is not None:
                df = df[cols]

        print('done, took {:.1f}s'.format(time.time() - s))
        return df, modified_cols

    def _build_columns(self, data, cols, type_casts, pg_cols):
        """Example args:

            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """
        print('Parsing...', end=' ')
        s = time.time()
        for col, typ in type_casts.items():
            if col not in data:
                continue
            if typ != np.datetime64:
                data[col] = data[col].astype(typ, copy=False)
            else:
                # Both infer_datetime_format and cache are critical for perf.
                data[col] = pd.to_datetime(data[col],
                                           infer_datetime_format=True,
                                           cache=True)

        # Discretize & create Columns.
        if cols is None:
            cols = data.columns
        columns = []
        if pg_cols is None:
            pg_cols = [None] * len(cols)
        for c, p in zip(cols, pg_cols):
            col = Column(c, pg_name=p)
            col.Fill(data[c])

            # dropna=False so that if NA/NaN is present in data,
            # all_distinct_values will capture it.
            #
            # For numeric: np.nan
            # For datetime: np.datetime64('NaT')
            col.SetDistribution(data[c].value_counts(dropna=False).index.values)
            columns.append(col)
        print('done, took {:.1f}s'.format(time.time() - s))

        return columns


class TableDataset(data.Dataset):
    """Wraps a Table and yields each row as a PyTorch Dataset element."""

    def __init__(self, table):
        super(TableDataset, self).__init__()
        self.table = copy.deepcopy(table)

        print('Discretizing table...', end=' ')
        s = time.time()
        # [cardianlity, num cols].
        self.tuples_np = np.stack(
            [self.Discretize(c) for c in self.table.Columns()], axis=1)
        self.tuples = torch.as_tensor(
            self.tuples_np.astype(np.float32, copy=False))
        print('done, took {:.1f}s'.format(time.time() - s))

    def Discretize(self, col):
        """Discretize values into its Column's bins.

        Args:
          col: the Column.
        Returns:
          col_data: discretized version; an np.ndarray of type np.int32.
        """
        return Discretize(col)

    def size(self):
        return len(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]


def Discretize(col, data=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.

    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data

    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan

        dvs = col.all_distinct_values[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        dvs = col.all_distinct_values
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    bin_ids = bin_ids.astype(np.int32, copy=False)
    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids
