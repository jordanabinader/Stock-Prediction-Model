import numpy as np
import pandas as pd
import datetime
from utils.str_to_datetime import str_to_datetime

def create_windowed_dataframe(dataframe, start_date_str, end_date_str, window_size=3):
    """
    Create a windowed dataframe from the input dataframe.

    Parameters:
    - dataframe: Input dataframe containing time series data.
    - start_date_str: Start date in string format (e.g., '2020-01-01').
    - end_date_str: End date in string format (e.g., '2023-12-31').
    - window_size: Size of the sliding window.

    Returns:
    - windowed_df: Windowed dataframe with columns for target date, window features, and target value.
    """
    start_date = str_to_datetime(start_date_str)
    end_date = str_to_datetime(end_date_str)

    target_date = start_date
    dates = []
    X, Y = [], []

    last_iteration = False

    while True:
        # Extract a window of data up to the target date
        df_subset = dataframe.loc[:target_date].tail(window_size + 1)

        if len(df_subset) != window_size + 1:
            print(f'Error: Window of size {window_size} is too large for date {target_date}')
            return

        values = df_subset['Returns'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        # Find the next date in the dataset within a week
        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = map(int, year_month_day)
        next_date = datetime.datetime(day=day, month=month, year=year)

        if last_iteration:
            break

        target_date = next_date

        if target_date == end_date:
            last_iteration = True

    # Create the windowed dataframe
    windowed_df = pd.DataFrame({})
    windowed_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, window_size):
        windowed_df[f'Feature-{window_size - i}'] = X[:, i]

    windowed_df['Target'] = Y

    return windowed_df
