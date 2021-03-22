import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt


def max_drawdown(date_series, capital_series):
    """
    计算最大回撤
    :param date_series: 日期序列
    :param capital_series: 收益序列
    :return:输出最大回撤及开始日期和结束日期
    """
    # 将数据序列合并为一个dataframe并按日期排序
    df = pd.DataFrame({'date': date_series, 'capital': capital_series})
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['max2here'] = pd.Series.expanding(df['capital']).max()
    df['dd2here'] = df['capital'] / df['max2here'] - 1  # 计算当日的回撤
    # 计算最大回撤和结束时间
    temp = df.sort_values(by='dd2here').iloc[0][['date', 'dd2here']]
    max_dd = temp['dd2here']
    end_date = temp['date']
    # 计算开始时间
    df = df[df['date'] <= end_date]
    start_date = df.sort_values(by='capital', ascending=False).iloc[0]['date']
    return max_dd, start_date, end_date


def annual_return(date_series, capital_series, num_work):
    """
    计算年化收益率
    :param date_series: 日期序列
    :param capital_series:资产序列
    :return: 输出在回测期间的年化收益率
    """
    df = pd.DataFrame({'date': date_series, 'capital': capital_series})
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # 计算年化收益率
    annual = pow(df.loc[len(df.index) - 1, 'capital'] / df.loc[0, 'capital'], num_work / len(capital_series)) - 1
    return annual


def volatility(date_series, return_series, num_work):
    """
    :param date_series: 日期序列
    :param return_series: 账户日收益率序列
    :return: 输出回测期间的收益波动率
    """
    df = pd.DataFrame({'date': date_series, 'rtn': return_series})
    # 计算波动率
    vol = np.log(df['rtn'] + 1).std() * np.sqrt(num_work)
    return vol


def sharpe_ratio(date_series, capital_series, return_series, num_work):
    """
    :param date_series: 日期序列
    :param capital_series: 账户价值序列
    :param return_series: 账户日收益率序列
    :return: 输出夏普比率
    """
    # 将数据序列合并为一个 dataframe 并按日期排序
    df = pd.DataFrame({'date': date_series, 'capital': capital_series, 'rtn': return_series})
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    rng = pd.period_range(df['date'].iloc[0], df['date'].iloc[-1], freq='D')

    rf = 0.0045  # 无风险利率取 10 年期国债的到期年化收益率
    # 账户年化收益
    annual_stock = annual_return(date_series, capital_series, num_work)
    # 计算收益波动率
    vol = volatility(date_series, return_series, num_work)
    # 计算夏普比
    sharpe = (annual_stock - rf) / vol
    if sharpe < -1000000:
        sharpe = -1000000
    return sharpe


def win_rate(return_series):
    """
    :param return_series: 账户日收益率序列
    :return: 胜率
    """
    total_days = len(return_series)
    v_loss_days = return_series.apply(lambda x: np.nan if x < 0 else 1).count()  # 亏损天数
    v_win_pro = (total_days - v_loss_days) / total_days
    return v_win_pro


def skewness(capital_series):
    """
    :param capital_series: 账户价值序列
    :return: 偏度
    """
    return capital_series.skew()


def kurtosis(capital_series):
    """
    :param capital_series: 账户价值序列
    :return: 峰度
    """
    return capital_series.kurt()


def draw_plot(df, isshow=True):
    fig, axis = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(15, 8)
    ax1 = axis[0]
    ax2 = axis[1]

    p1, = ax1.plot(df.index, df['strategy'], color='steelblue', linewidth=2)
    p2, = ax1.plot(df.index, df['benchmark'], color='r', linewidth=2)

    title1 = 'Cumulative Return from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax1.set_title(title1)
    ax1.legend([p1, p2], ['strategy', 'benchmark'])
    ax1.grid(True)
    ax2.set_xlim(df.index[0], df.index[-1])

    f1 = ax2.fill_between(df.index, df['MaxDD_strategy'], 0, facecolor='steelblue', alpha=0.5)
    f2 = ax2.fill_between(df.index, df['MaxDD_benchmark'], 0, facecolor='r', alpha=0.5)

    title2 = 'Drawdown from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax2.legend([f1, f2], ['MaxDD_strategy', 'MaxDD_benchmark'], loc=3)
    ax2.set_title(title2)
    ax2.grid(True)
    ax2.set_xlim(df.index[0], df.index[-1])
    if isshow == True:
        plt.show()



