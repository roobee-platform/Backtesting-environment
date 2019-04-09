from pandas_datareader import data
import pandas as pd
import os


def get_data(ticker, source, start_t, end_t):
    """
    The function loads datasets from yahoo finance/internal data storage and processes it into convinient OHLC format
    :param ticker: string, a ticker name or name of a file to load
    :param source: string, an option to load from yahoo finance or a path to data storage
    :param start_t: string/datetime
    :param end_t: string/datetime
    :return: pd.Dataframe
    """
    if source == 'yahoo':
        OHLC = data.DataReader(ticker, 'yahoo', start_t, end_t)
    else:
        files = os.listdir(source)
        file = [x for x in files if ticker in x][0]
        OHLC = pd.read_csv(os.path.join(source, file), delimiter=';', index_col=0, header=None)  # .iloc[:, :4]
        OHLC.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap']
        OHLC.index = pd.to_datetime(OHLC.index)
        OHLC = OHLC[(OHLC.index <= end_t) & (OHLC.index > start_t)]
        OHLC = OHLC.fillna(method='ffill')
        OHLC = OHLC.iloc[::-1]

    return OHLC


def get_close_dataset(tickers, source, start_t, end_t, fields=['Close']):
    """
    Loads processed OHLC dataframes and processes them into a dataset with selected fields from OHLC
    :param tickers: list, e.g: ['XXX','XXY']
    :param source: string, an option to load from yahoo finance or a path to data storage
    :param start_t: string/datetime
    :param end_t: string/datetime
    :param fields: fields from OHLC to load
    :return:
    """
    closes = pd.DataFrame()
    for ticker in tickers:
        # closes[ticker] = get_data(ticker, source, start_t, end_t)['Close']
        newcol = get_data(ticker, source, start_t, end_t)[fields]
        newcol.columns = [x+'_'+ticker for x in fields]
        closes = closes.join(newcol, how='outer')

    closes.astype(float)
    return closes
