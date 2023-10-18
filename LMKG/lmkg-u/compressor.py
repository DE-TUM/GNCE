"""
    Compression of columns that have large number of unique domain values
"""

class Compressor:
    def __init__(self, root):
        '''
        Class that splits a column into two 'root' separate columns
        :param root: the number of columns that should be created
        '''
        self.split_columns_index = set()
        self.split_columns_dividers = dict()
        self.root = root

    def divide_column(self, column_values, column_divider, original_col_index):
        '''
        Method for splitting the column value based on the largest root number.
        :param column_values: all the values from the column
        :param col_name: the column title
        :param column_divider: the number which will be used for division
        :return:
        '''
        quotient_column = []
        reminder_column = []

        self.split_columns_index.add(original_col_index)
        self.split_columns_dividers[original_col_index] = column_divider

        for val in column_values:
            # divide the value with the largest root number and get the integer of the result
            # add + 1 to avoid the 0
            multiplier = int(int(val) / column_divider)
            quotient_column.append((multiplier + 1))

            # get the reminder from the division and increase by 1 to avoid 0
            reminder = abs(int(val) - (multiplier * column_divider))
            reminder_column.append((reminder + 1))

        return quotient_column, reminder_column

    def split_single_value_for_column(self, column_value, column_index):
        column_divider = self.split_columns_dividers[column_index]
        # get the quotient and sum it up with 1 (to avoid 0)
        multiplier = int(int(column_value) / column_divider) + 1
        # get the reminder and sum it up with 1 (to avoid 0)
        reminder = abs(int(column_value) - ((multiplier - 1) * column_divider)) + 1

        return multiplier, reminder

    def return_original_value_when_column_split_in_two(self, reminder_column_value, quotient_column_value, original_column_id):
        original_value = (quotient_column_value-1)*self.split_columns_dividers[original_column_id] + (reminder_column_value-1)
        return original_value


