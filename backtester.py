import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from data_read import get_close_dataset


def get_dirichlet_restrictions(N):
    difarr = np.random.randint(2, 10, N)
    return np.random.dirichlet(difarr)


def generate_n_random_portfolios(returns_period, covariance, num_portfolios=20000):
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

    num_assets = len(returns_period.dropna())
    num_portfolios = num_portfolios
    zerowghts = returns_period.index[pd.isna(returns_period)]
    positivewghts = returns_period.index[[not i for i in pd.isna(returns_period)]]
    print('positivewghts', positivewghts)
    print('zerowghts', zerowghts)
    print('num_assets', num_assets)

    if len(positivewghts) != 1:
        for single_portfolio in range(num_portfolios):
            weights = get_dirichlet_restrictions(num_assets)
            returns_daily = np.dot(weights, returns_period.dropna())  # how much we get from portfolio
            volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            sharpe = returns_daily / volatility  # sharpe ratio
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
        # now store all the results
        sharpe_ratio.append(float(sharpe))
        port_returns.append(float(returns_daily))
        port_volatility.append(float(volatility))
        stock_weights.append([weights])

    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter, symbol in enumerate(positivewghts):
        portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]
    for counter, symbol in enumerate(zerowghts):
        portfolio[symbol+' Weight'] = 0

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)
    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock +' Weight' for stock in returns_period.index]
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
    portfolio = stds/stds.sum(axis=1).values[0]
    portfolio.columns = [x+'_Weight' for x in portfolio.columns]
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


# define time range
start_date = '2015-12-31'
end_date = '2019-04-01'

# Tickers in portfolio
tickers = [
    'AAPL',
    'AMZN',
    'NVDA',
    'NFLX',
    'MSFT'
]

# define rebalancing method
# Variants are: 'Mean-Variance', 'Volume weighted', 'Inverse Volatility' and 'Custom'
alpha = 'Mean-Variance'
alpha = 'Volume weighted'
# optimization parameter for Mean-Variance method: 'Sharpe', 'Min Variance',
parameter = 'Sharpe'

# define rebalancing period in months
rebalancing_period = 6

# load required dataset
closes = get_close_dataset(tickers, source='yahoo',
                           start_t=start_date, end_t=end_date,
                           fields=['Close', 'Volume'])
volumes = closes[[x for x in closes.columns if 'Volume' in x]]
closes = closes[[x for x in closes.columns if 'Close' in x]]


# all calculations begin here
# these store data
pnl_target = []
pnl_months = []
start = start_date
for end in pd.date_range(start=start_date, end=end_date, freq=str(rebalancing_period)+'M', closed='right'):
    print(start, end)
    # start = end
    close_prices = closes[(closes.index <= end) & (closes.index >= start)]
    # volumes_assets = volumes[(volumes.index <= end) & (volumes.index >= start)]
    idf = close_prices.index[0]
    ind = closes.index.get_loc(idf)
    if ind == 0:
        ind = 1
    close_prices = closes.iloc[ind-1:, :]
    close_prices = close_prices[close_prices.index <= end]
    close_prices.fillna(method='ffill', inplace=True)
    try:
        returns_period = (close_prices.iloc[-1, :] - close_prices.iloc[0, :]) / close_prices.iloc[0, :]
    except:
        start = end
        continue

    print('returns: \n{}'.format(returns_period))

    if alpha == 'Mean-Variance':
        covariance_matrix = close_prices[returns_period.dropna().index].dropna(axis=1).cov()
        df = generate_n_random_portfolios(returns_period,
                                          covariance_matrix,
                                          num_portfolios=20000)

        min_volatility = df["Volatility"].min()
        max_sharpe = df['Sharpe Ratio'].max()
        # use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_port = df.loc[df['Volatility'] == min_volatility]
        if parameter == 'Sharpe':
            target_portfolio = sharpe_portfolio
        if parameter == 'Min Variance':
            target_portfolio = min_variance_port
    if alpha == 'Volume weighted':
        df = generate_volume_weights(volumes, end)
        target_portfolio = df
    # if alpha == 'Volume weighted':

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
            month_ret = np.dot(wgths, m_returns.fillna(0))[0]
            pnl_target.append(month_ret+1)
            pnl_months.append(i.month_name()[:3]+' '+str(i.year)[2:])
        except IndexError:
            continue

    start = end

portfolio_returns = pd.DataFrame(data=pnl_target, index=pnl_months)
portfolio_returns.columns = ['Portfolio_returns']
portfolio_returns['CumRets'] = portfolio_returns['Portfolio_returns'].cumprod()  # * 10
portfolio_returns['CumRets'].plot()

print('Done')
