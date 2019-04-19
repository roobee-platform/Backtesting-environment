import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from data_read import get_close_dataset
import open_source_backtester.preset_builder as psb
# import random
# random.seed(124)


def get_dirichlet_restrictions(number_of_tokens, tokens):
    while True:
        result = {}
        difarr = np.random.randint(2, 10, number_of_tokens)
        rates = np.random.dirichlet(difarr)
        i = 0
        for token in tokens:
            result[token] = round(rates[i], 3)
            i = i + 1
        limits = 1
        sum = 0
        for key in result:
            sum += result[key]
            if result[key] > 0.65:
                limits = 0
                break
            if result[key] < 0.03:
                limits = 0
                break
        if limits == 1 and sum == 1.0:
            result = pd.Series(list(result.values()))
            return result


def generate_n_random_portfolios(returns_period,
                                 covariance,
                                 num_portfolios=20000,
                                 *kwargs):
    """
    Generate N random portfolios and calculate their pnl/stats for a given period
    :param returns_period: list or pd.Series with returns for a given data
    :param covariance: a covariance matrix for a given period
    :param num_portfolios: integer, number of random portfolios to generate
    :return:
    """
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []
    inform_ratio = []

    # if 'benchmark_return' in kwargs[0].keys():

    num_assets = len(returns_period.dropna())
    num_portfolios = num_portfolios
    zerowghts = returns_period.index[pd.isna(returns_period)]
    positivewghts = returns_period.index[[not i for i in pd.isna(returns_period)]]
    print('positivewghts', positivewghts)
    print('zerowghts', zerowghts)
    print('num_assets', num_assets)

    if len(positivewghts) != 1:
        for single_portfolio in range(num_portfolios):
            weights = get_dirichlet_restrictions(number_of_tokens=num_assets, tokens=tickers)
            returns_daily = np.dot(weights, returns_period.dropna())  # how much we get from portfolio
            volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            sharpe = returns_daily / volatility  # sharpe ratio
            if 'benchmark_return' in kwargs[0].keys():
                portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
                info = (returns_daily - kwargs[0]['benchmark_return']) / \
                       (portfolio_variance + kwargs[0]['benchmark_var'])
                inform_ratio.append(info.values[0])
            else:
                inform_ratio.append(np.nan)
            # now store all the results
            sharpe_ratio.append(sharpe)
            port_returns.append(returns_daily)
            port_volatility.append(volatility)
            stock_weights.append(weights)
    else:
        weights = 1
        returns_daily = np.dot(weights, returns_period.dropna())  # how much we get from portfolio
        volatility = np.sqrt(np.dot(weights, np.dot(covariance, weights)))  # weighted variance
        sharpe = returns_daily / volatility  # sharpe ratio
        if 'benchmark_return' in kwargs[0].keys():
            portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
            info = (returns_daily - kwargs[0]['benchmark_return']) / \
                   (portfolio_variance + kwargs[0]['benchmark_var'])
            inform_ratio.append(info.values)
        else:
            inform_ratio.append(np.nan)

        # now store all the results
        sharpe_ratio.append(float(sharpe))
        port_returns.append(float(returns_daily))
        port_volatility.append(float(volatility))
        stock_weights.append([weights])

    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio,
                 'Information Ratio': inform_ratio}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter, symbol in enumerate(positivewghts):
        portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]
    for counter, symbol in enumerate(zerowghts):
        portfolio[symbol+' Weight'] = 0

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)
    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio', 'Information Ratio'] +\
                   [stock + ' Weight' for stock in returns_period.index]
    # reorder dataframe columns
    df = df[column_order]

    return df


def generate_volume_weights(volumes, date):
    time_delta = datetime.timedelta(days=0)
    while True:
        try:
            volumes_period = volumes[(volumes.index == date)]
            portfolio = volumes_period/volumes_period.sum(axis=1).values[0]
            break
        except IndexError:
           time_delta += datetime.timedelta(days=2)
           volumes_period = volumes[(volumes.index == date-time_delta)]
           break
    portfolio = volumes_period/volumes_period.sum(axis=1).values[0]
    for i in portfolio.values[0]:
        if i < 0.02:
            addname = portfolio[portfolio != 0]
            name = addname.index[addname == addname.min()]
            mval = portfolio[portfolio < 0.02]
            mval = mval[mval != 0]
            name_mval = portfolio.index[portfolio == mval.max()][0]
            if mval.shape[0] > 1:
                portfolio[name_mval] += portfolio.min()
                portfolio[i] = 0
            else:
                # add weight to the least one but more that this
                minval = portfolio[[x for x in portfolio.index if x not in name]].min()
                name_mval = portfolio.index[portfolio == minval][0]
                portfolio[name_mval] += portfolio.min()
                portfolio[i] = 0
    time_delta = 0
    portfolio.columns = [x+'_Weight' for x in portfolio.columns]
    return portfolio


def generate_inverse_volatility_weights(returns):
    stds = returns.std()
    portfolio = stds/stds.sum()

    portfolio.index = [x+'_Weight' for x in portfolio.index]
    portfolio = pd.DataFrame(portfolio).T
    return portfolio


def visualize_frontier(df):
    """
    Visualize profit-loss of each randomly generated portfolio in terms of return/variance
    :param df: A dataframe with random portfolio weights and sharpe/volatility data
    :return:
    """
    min_volatility = df["Volatility"].min()
    max_sharpe = df['Sharpe Ratio'].max()
    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
    plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.show()
    return True


def process_returns(pnl_target, pnl_months, plot=False):
    portfolio_returns = pd.DataFrame(data=pnl_target, index=pnl_months)
    portfolio_returns.columns = ['Portfolio_returns']
    portfolio_returns['CumRets'] = portfolio_returns['Portfolio_returns'].cumprod()  # * 10
    if plot == 1:
        (portfolio_returns['CumRets']-1).plot()
    return portfolio_returns


# define time range
start_date = '2016-12-31'
end_date = '2019-04-01'

name = 'CryptoTransactionCoins'

# Tickers in portfolio
tickers = [
    'bitcoin.csv',
    'ripple',
    'stellar.csv',
    'litecoin.csv',
    'monero.csv',
    'zcash.csv',
    'dash.csv'
]

source = 'datasets/data'

optimization = 'min_variance_port'

# define rebalancing method
# Variants are: 'Mean-Variance', 'Volume weighted', 'Inverse Volatility' and 'Custom'
alpha = 'Mean-Variance'
# alpha = 'Volume weighted'
# alpha = 'Inverse Volatility'
# optimization parameter for Mean-Variance method: 'Sharpe', 'Min Variance', 'Information'
parameter = 'Min Variance'
if parameter == 'Information':
    benchmark_ticker = ['DEF']

# define rebalancing period in months
rebalancing_period = 1

# load required dataset
closes = get_close_dataset(tickers, source=source,
                           start_t=start_date, end_t=end_date,
                           fields=['Close', 'Volume'])
volumes = closes[[x for x in closes.columns if 'Volume' in x]]
closes = closes[[x for x in closes.columns if 'Close' in x]]
returns = closes.pct_change().fillna(0)
returns.columns = [x+'_Returns' for x in tickers]

benchmark_data = dict()
if parameter == 'Information':
    benchmark = get_close_dataset(benchmark_ticker, source='yahoo',
                                  start_t=start_date, end_t=end_date,
                                  fields=['Close'])

# all calculations begin here
# these store data
pnl_target = []
pnl_months = []
hist_weights = []
hist_rets = []
start = start_date
for end in pd.date_range(start=start_date, end=end_date, freq=str(rebalancing_period)+'M', closed='right'):
    print(start, end)
    # start = end
    close_prices = closes[(closes.index <= end) & (closes.index >= start)]
    if parameter == 'Information':
        benchmark_data['benchmark_prices'] = benchmark[(benchmark.index <= end) & (benchmark.index >= start)]
    # volumes_assets = volumes[(volumes.index <= end) & (volumes.index >= start)]

    try:
        idf = close_prices.index[0]
    except NameError:
        start = end
        continue

    ind = closes.index.get_loc(idf)
    if ind == 0:
        ind = 1
    close_prices = closes.iloc[ind-1:, :]
    close_prices = close_prices[close_prices.index <= end]
    close_prices.fillna(method='ffill', inplace=True)
    try:
        returns_period = (close_prices.iloc[-1, :] - close_prices.iloc[0, :]) / close_prices.iloc[0, :]
        if parameter == 'Information':
            benchmark_data['returns_period_benchmark'] = (benchmark_data['benchmark_prices'].iloc[-1, :] -
                                                           benchmark_data['benchmark_prices'].iloc[0, :]) / \
                                                          benchmark_data['benchmark_prices'].iloc[0, :]

    except:
        start = end
        continue

    print('returns: \n{}'.format(returns_period))

    if alpha == 'Mean-Variance':
        covariance_matrix = close_prices[returns_period.dropna().index].dropna(axis=1).cov()
        if parameter == 'Information':
            benchmark_data['bmrk_variance'] = benchmark_data['benchmark_prices'].var().values[0]
        # else:
        #     bmrk_variance = 'None'
        df = generate_n_random_portfolios(returns_period,
                                          covariance_matrix,
                                          10000,
                                          benchmark_data)

        min_volatility = df["Volatility"].min()
        max_sharpe = df['Sharpe Ratio'].max()
        max_info = df['Information Ratio'].max()
        # use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_port = df.loc[df['Volatility'] == min_volatility]
        max_info_port = df.loc[df['Information Ratio'] == max_info]
        if parameter == 'Sharpe':
            target_portfolio = sharpe_portfolio
        if parameter == 'Min Variance':
            target_portfolio = min_variance_port
        if parameter == 'Information':
            target_portfolio = max_info_port
    if alpha == 'Volume weighted':
        df = generate_volume_weights(volumes, end)
        target_portfolio = df
    if alpha == 'Inverse Volatility':
        df = generate_inverse_volatility_weights(close_prices.pct_change().dropna())
        target_portfolio = df

    y_weights = [x for x in target_portfolio.columns if 'Weight' in x]
    # we must shift data by 1 period (initially we calculate data for the period that has passed)
    start_bt = end
    end_bt = pd.Timestamp(end) + pd.DateOffset(months=rebalancing_period)
    for i in pd.date_range(start=start_bt, end=end_bt, freq='BM'):
        try:
            close_prices = closes[(closes.index <= end_bt) & (closes.index > start_bt)]
            m_prices = close_prices[close_prices.index.month == i.month]
            idf = m_prices.index[0]
            ind = close_prices.index.get_loc(idf)
            if ind == 0:
                ind = 1
            end_prev_period = close_prices.iloc[ind-1, :]
            # end_prev_period.append(m_prices)
            m_returns = (m_prices.iloc[-1, :] - end_prev_period) / end_prev_period
            wgths = target_portfolio[y_weights]

            hist_weights.append(list(wgths.values[0]))
            hist_rets.append(m_returns.values)

            month_ret = np.dot(wgths, m_returns.fillna(0))[0]
            pnl_target.append(month_ret+1)
            pnl_months.append(i.month_name()[:3]+' '+str(i.year))
        except IndexError:
            continue

    start = end

portfolio_returns = process_returns(pnl_target, pnl_months)  # for portfolio

weights_history = pd.DataFrame(data=hist_weights, index=pnl_months)
weights_history.columns = [x[6:] for x in target_portfolio.columns if 'Weight' in x]
if returns.index[-1] != pd.to_datetime(weights_history.index[-1]):
    weights_history = pd.concat([weights_history, pd.DataFrame([weights_history.iloc[-1, :]],
                                                               index=[returns.index[-1]])])
weights_history.index = pd.to_datetime(weights_history.index)
weights_history = weights_history.resample('1D').ffill()
# weights_history = weights_history.shift(1).dropna(0)

# together = closes.join(returns).join(weights_history)#.dropna()
together = returns.join(weights_history)
together.dropna(thresh=weights_history.shape[1]+1, inplace=True)
together = together.join(closes)
for i in range(len(together)):
    if together.index[i].weekday() == 0:
        together = together[together.index >= together.index[i]]
        break

together = together[[x for x in together.columns if 'Close' in x] +
                    [x for x in together.columns if 'Returns' in x] +
                    [x for x in together.columns if 'Weight' in x]]

together.to_excel('historical_data/'+name+'.xls')
together = pd.read_excel('historical_data/'+name+'.xls', index_col=0, parse_dates=True)

weights_together = together[[x for x in together.columns if 'Weight' in x]]
returns_together = together[[x for x in together.columns if 'Returns' in x]]
returns_together.iloc[0, :] = 0
portfolio_returns = returns_together * weights_together.values
portfolio_returns = portfolio_returns.sum(axis=1)
Capital = 1000000
Capital1 = pd.DataFrame(portfolio_returns, index=returns_together.index, columns=['Returns'])
Capital1['priceChangeTotal'] = (Capital1['Returns']+1).cumprod()
Capital1['costPerUnit'] = Capital*Capital1['priceChangeTotal']
Capital1['priceChangeTotal'] = Capital1['priceChangeTotal']-1
Capital1_weekly = Capital1[Capital1.index.weekday == 0]
Capital1_weekly['priceChangePerWeek'] = Capital1_weekly['$costPerUnit'].pct_change().fillna(0)
del Capital1_weekly['Returns']
Capital1_weekly = Capital1_weekly[['costPerUnit', 'priceChangeTotal', 'priceChangePerWeek']]
Capital1_weekly.index.name = 'date'
Capital1_weekly.to_csv('historical_returns/'+name+'_weekly.csv')
Capital1.to_excel('historical_returns/'+name+'_daily.xls')

print('Done')
