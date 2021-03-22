import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import cvxpy as cp
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import performance as pf
from scipy.optimize import minimize
import os

warnings.filterwarnings('ignore')
plt.style.use('bmh')
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data\\数据汇总1.0.xlsx')


def best_match(df_corr, method=0):
    """
    资产配对
    :param df_corr:协方差矩阵[dataframe]
    :param method: 配对方法选择{0：经济意义；1：统计意义}[bool]
    :return: 配对资产与未配对资产[list]
    """
    # 资产匹配
    assets = df_corr.columns.to_list()
    n = len(assets)
    if method == 0:
        match_list = [['沪深300', '中国10年期国债期货'], ['SPX', '美国10年期国债期货'], ['日经225', '日本10年期国债期货'],
                     ['德国DAX', '德国10年期国债期货'], ['英国富时100', '英国10年期国债期货'], ['美元指数', 'BCOMPR'],
                     ['BCOMEN', '美国5年期国债期货'], ['BCOMIN', '德国5年期国债期货']]
        unmatch_list = ['中国5年期国债期货']
        return match_list, unmatch_list
    if method == 1:
        corr_ = df_corr.values.flatten()

        x = cp.Variable((n, n), integer=True)
        cons1 = [np.ones(n) @ x[i] <= 1 for i in range(n)]
        cons2 = [np.ones(n) @ x[:, i] <= 1 for i in range(n)]
        cons3 = [x[i, j] == x[j, i] for i in range(n) for j in range(n)]

        cons = [x >= 0, x <= 1] + cons1 + cons2 + cons3
        func = cp.Minimize(x.flatten() @ corr_)
        prob = cp.Problem(func, cons)
        prob.solve()
        x_sol = np.triu(x.value)
        loc_match = [[i, j] for i in range(n) for j in range(n) if x_sol[i, j] == 1]
        loc_unmatch = [i for i in range(n) if x.value.sum(axis=0)[i] == 0]

        match_list = []
        for m in loc_match:
            match_list.append([assets[m[0]], assets[m[1]]])
        unmatch_list = [assets[i] for i in loc_unmatch]
        print(match_list, unmatch_list)

    return match_list, unmatch_list


def risk_budget(stockret, rb, isshort=False):
    """
    风险预算——rb为等权矩阵时即为风险平价
    :param stockret:资产收益率[dataframe]
    :param rb:资产风险贡献矩阵[array]
    :param isshort:是否允许卖空[bool]
    :return:资产配置权重[array]
    """
    covij = stockret.cov()
    # 初始权重
    w0 = 1 / stockret.std(axis=0).values
    w0 = w0 / w0.sum()
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    if not isshort:
        cons = cons + [{'type': 'ineq', 'fun': lambda x: x}]
    func = lambda x: np.sum((x * np.dot(covij, x) / np.dot(x.T, np.dot(covij, x)) - rb) ** 2)

    res = minimize(func, w0, method='SLSQP', constraints=cons, tol=1e-10)
    weight = res.x
    return weight


def initial_pos(return_df, method=0, isshort=False):
    """
    初始权重的确定——运用风险预算或风险平价算法
    :param return_df: 资产收益率[dataframe]
    :param method: 算法选择{0：风险预算；1：风险平价}[bool]
    :return: 资产配置权重[dataframe]
    """
    num_asset = len(return_df.columns)
    # 风险预算
    if method == 0:
        vol_df = return_df.shift(1).rolling(250).std() * np.sqrt(250)
        weight_df = 0.05 / vol_df
        weight_df = weight_df.resample('M').first()
        weight_rb = weight_df.dropna().copy()
    # 风险平价
    if method == 1:
        weight_df = return_df.resample('M').first()
        weight_df = weight_df[1:]
        date = weight_df.index
        for d in tqdm(date):
            sub_df = return_df.loc[:str(d - relativedelta(months=1) + relativedelta(days=1))][:250]
            if len(sub_df) < 100:
                weight_df.loc[d] = np.nan
                continue
            rb = [1. / num_asset] * num_asset
            w = risk_budget(sub_df, rb, isshort)
            weight_df.loc[d] = w
        weight_rb = weight_df.dropna().copy()
    return weight_df, weight_rb


def momentum_adjust(return_df, weight_df, match_list, unmatch_list):
    """
    动量效应调整权重
    :param return_df:资产收益率[dataframe]
    :param weight_df:资产初始配置权重[dataframe]
    :param match_list:配对资产[list]
    :param unmatch_list:未配对资产[list]
    :return:资产调整后配置权重[dataframe]
    """
    return_df_monthly = return_df.resample('M').mean()
    adjust_weight = return_df_monthly

    for m in match_list:
        # r：风险资产; rf：防御资产
        r, rf = m[0], m[1]
        adjust_weight[r] = np.where(adjust_weight[r] > adjust_weight[rf], 1, 0)
        adjust_weight[r] = 0.25 + 0.5 * adjust_weight[r].shift(1).rolling(12).mean()
        adjust_weight[rf] = 1 - adjust_weight[r]
    for um in unmatch_list:
        adjust_weight[um] = np.where(adjust_weight[um] > 0, 1, 0)
        adjust_weight[um] = 0.25 + 0.5 * adjust_weight[um].shift(1).rolling(12).mean()

    weight_df = weight_df * adjust_weight
    weight_df = weight_df.dropna(how='all')
    return weight_df


def strategy_select(return_df, weight_df, match_list, unmatch_list, method):
    """
    动量效应/VIX效应调整权重
    :param return_df:资产收益率[dataframe]
    :param weight_df:资产初始配置权重[dataframe]
    :param match_list:配对资产[list]
    :param unmatch_list:未配对资产[list]
    :param method:策略方法{0：动量效应；1：VIX效应}[bool]
    :return:资产调整后配置权重[dataframe]
    """
    return_df_monthly = return_df.resample('M').mean()
    adjust_weight = return_df_monthly

    # 动量效应
    if method == 0:
        for m in match_list:
            # r：风险资产; rf：防御资产
            r, rf = m[0], m[1]
            adjust_weight[r] = np.where(adjust_weight[r] > adjust_weight[rf], 1, 0)
            adjust_weight[r] = 0.25 + 0.5 * adjust_weight[r].shift(1).rolling(12).mean()
            adjust_weight[rf] = 1 - adjust_weight[r]
        for um in unmatch_list:
            adjust_weight[um] = np.where(adjust_weight[um] > 0, 1, 0)
            adjust_weight[um] = 0.25 + 0.5 * adjust_weight[um].shift(1).rolling(12).mean()

    # VIX效应
    if method == 1:
        vix_df = pd.read_csv(r"data/VIX_position.csv")
        for i in np.arange(250, len(vix_df)):
            month_vix = vix_df.loc[i - 30:i - 1, 'VIX'].mean()
            year_vix = vix_df.loc[i - 250:i - 1, 'VIX']
            vix_df.loc[i, 'percentile'] = len(year_vix[year_vix >= month_vix]) / len(year_vix)
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])
        datearr = np.union1d(vix_df.Date, return_df_monthly.index)
        vix_df = vix_df.set_index('Date').reindex(datearr).sort_index(ascending=True). \
            fillna(method='ffill').reindex(return_df_monthly.index)

        for m in match_list:
            r, rf = m[0], m[1]
            adjust_weight[r] = vix_df['percentile']
            adjust_weight[rf] = 1 - adjust_weight[r]
        for um in unmatch_list:
            adjust_weight[um] = vix_df['percentile']

    weight_df = weight_df * adjust_weight
    weight_df = weight_df.dropna(how='all')

    return weight_df


def back_test(return_df, weight_m, weight_rb):
    """
    风险调整与动量效应的回测
    :param return_df:资产收益率[dataframe]
    :param weight_m:动量效应配置权重[dataframe]
    :param weight_rb:风险调整配置权重[dataframe]
    :return: 指数收益率[dataframe]、杠杆率[dataframe]
    """
    months = list(np.unique([m[:-3] for m in weight_m.index.astype(str).to_list()]))
    index_df = pd.DataFrame()
    leverage_df = pd.DataFrame(columns=return_df.columns)
    for m in months:
        # 风险调整后的收益率序列
        sub_df1 = (((return_df[m] + 1).cumprod() * weight_rb[m].values).sum(axis=1)).pct_change().fillna(0)
        sub_df1 = pd.DataFrame(sub_df1).rename(columns={0: 'return_risk_budget'})

        # 动量调整后的收益率序列和权重
        sub_lever = (return_df[m] + 1).cumprod() * weight_m[m].values
        sub_df0 = sub_lever.copy().sum(axis=1)
        sub_lever = sub_lever.div(sub_lever.sum(axis=1), axis=0)
        sub_df0 = pd.DataFrame(sub_df0.pct_change().fillna(0), columns=['return_momentum'])

        sub_df = pd.concat([sub_df0, sub_df1], axis=1)
        index_df = pd.concat([index_df, sub_df])
        leverage_df = pd.concat([leverage_df, sub_lever])
    return index_df, leverage_df


def volatility_adjust(index_df, return_df, leverage_df, sigma_level):
    """
    波动率控制
    :param index_df:回测指数序列[dataframe]
    :param return_df:资产收益率[dataframe]
    :param leverage_df:杠杆率[dataframe]
    :param sigma_level:波动率控制水平[float]
    :return:回测指数序列[dataframe]、杠杆率[dataframe]
    """
    index_df['vol'] = np.log(index_df['return_momentum'] + 1).shift(1).rolling(250).std() * np.sqrt(250)
    index_df['proportion'] = (sigma_level / index_df['vol']).fillna(1)

    leverage_df = leverage_df.multiply(index_df['proportion'], axis='index')

    index_df['return_vol_control'] = (leverage_df * return_df).sum(axis=1)

    return index_df, leverage_df


def vix_adjust(index_df, vix_df, leverage_df, equity):
    """
    vix调整
    :param index_df:回测指数序列[dataframe]
    :param vix_df:vix序列[dataframe]
    :param leverage_df:杠杆率[dataframe]
    :param equity:风险资产[list]
    :return:回测指数序列[dataframe]
    """
    bond = [i for i in leverage_df.columns if i not in equity]
    leverage_df['equity_weight'] = leverage_df[equity].sum(axis=1)
    leverage_df['bond_weight'] = leverage_df[bond].sum(axis=1)
    index_df = pd.concat([index_df, vix_df['vix_return_rate'], leverage_df['equity_weight']], axis=1, join='inner')
    index_df['return_vix_control'] = index_df['return_vol_control'] + 0.05 * index_df['vix_return_rate'] * \
                                     index_df['equity_weight']
    return index_df, leverage_df


def cal_index_param_series(index_df):
    """
    计算各策略回测后的净值和回撤率序列
    :param index_df:回测指数序列[dataframe]
    :return:回测指数序列[dataframe]
    """
    # 风险调整的净值、回撤率序列
    index_df['capital_risk_budget'] = (index_df['return_risk_budget'] + 1).cumprod()
    index_df['MaxDD_risk_budget'] = index_df['capital_risk_budget'] / index_df['capital_risk_budget'].cummax() - 1

    # 动量调整的净值、回撤率序列
    index_df['capital_momentum'] = (index_df['return_momentum'] + 1).cumprod()
    index_df['MaxDD_momentum'] = index_df['capital_momentum'] / index_df['capital_momentum'].cummax() - 1

    # 波动率控制的净值、回撤率序列
    index_df['capital_vol_control'] = (index_df['return_vol_control'] + 1).cumprod()
    index_df['MaxDD_vol_control'] = index_df['capital_vol_control'] / index_df['capital_vol_control'].cummax() - 1

    # vix控制的净值、回撤率序列
    index_df['capital_vix_control'] = (index_df['return_vix_control'] + 1).cumprod()
    index_df['MaxDD_vix_control'] = index_df['capital_vix_control'] / index_df['capital_vix_control'].cummax() - 1

    return index_df


def cal_risk_nonrisky_return(return_df, leverage_df, equity):
    bond = [i for i in return_df.columns if i not in equity]
    
    if len(return_df) < len(leverage_df):
        index = return_df.index
    else:
        index = leverage_df.index
    return_df = return_df.reindex(index)
    leverage_df = leverage_df.reindex(index)
    df = return_df * leverage_df[return_df.columns]
    df['风险资产'] = (df[equity].sum(axis=1) + 1).cumprod()
    df['防御资产'] = (df[bond].sum(axis=1) + 1).cumprod()
    df['MaxDD_rf'] = df['风险资产'] / df['风险资产'].cummax() - 1
    df['MaxDD_risky'] = df['防御资产'] / df['防御资产'].cummax() - 1
    
    months = list(np.unique([m[:-3] for m in df.index.astype(str).to_list()]))
    month_return = pd.DataFrame(columns=['风险资产', '防御资产'])
    for m in months:
        sub_df = df[m]
        month_return.loc[m, '风险资产'] = sub_df['风险资产'][-1] / sub_df['风险资产'][0] - 1
        month_return.loc[m, '防御资产'] = sub_df['防御资产'][-1] / sub_df['防御资产'][0] - 1
    month_return.to_excel('month_return.xlsx')
    return df


def cal_indicator(date, return_series, capital, ydays):
    """
    指标计算
    :param date:日期序列[series]
    :param return_series:资产收益率[series]
    :param capital:资产收益率[series]
    :param ydays:年工作日数[int]
    :return: 年化收益率、年化波动率、最大回撤、夏普比率、胜率、偏度、峰度[float]
    """
    cumu_return = round(capital[-1] - 1, 3)
    annual_r = round(pf.annual_return(date, capital, ydays), 3)
    annual_vol = round(pf.volatility(date, return_series, ydays), 3)
    sharpe = round(pf.sharpe_ratio(date, capital, return_series, ydays), 3)
    maxdd, _, _ = pf.max_drawdown(date, capital)
    success_rate = round(pf.win_rate(return_series), 3)
    skew = round(pf.skewness(capital), 3)
    kurt = round(pf.kurtosis(capital), 3)
    return cumu_return, annual_r, annual_vol, -round(maxdd, 3), sharpe, success_rate, skew, kurt


def backtest_perform(index_df, strategy):
    """
    指标计算
    strategy:{'VIX', '动量'}
    """
    risk_budget_indicators = cal_indicator(index_df.index, index_df['return_risk_budget'],
                                           index_df['capital_risk_budget'], 250)
    monmentum_indicators = cal_indicator(index_df.index, index_df['return_momentum'],
                                         index_df['capital_momentum'], 250)
    vol_control_indicators = cal_indicator(index_df.index, index_df['return_vol_control'],
                                           index_df['capital_vol_control'], 250)
    vix_control_indicators = cal_indicator(index_df.index, index_df['return_vix_control'],
                                           index_df['capital_vix_control'], 250)
    print('风险预算【累计收益率，年化收益率，年化波动率，最大回撤，夏普比率，胜率，偏度，峰度】：{}'.format(risk_budget_indicators))
    print(strategy + '效应【累计收益率，年化收益率，年化波动率，最大回撤，夏普比率，胜率，偏度，峰度】：{}'.format(monmentum_indicators))
    print('波动率控制【累计收益率，年化收益率，年化波动率，最大回撤，夏普比率，胜率，偏度，峰度】：{}'.format(vol_control_indicators))
    print('VIX控制【累计收益率，年化收益率，年化波动率，最大回撤，夏普比率，胜率，偏度，峰度】：{}'.format(vix_control_indicators))


def draw_plot(df, strategy='momentum_strategy', isshow=True):
    """
    绘制回测曲线图
    strategy:{'vix_strategy', 'momentum_strategy'}
    """
    fig, axis = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(15, 8)
    ax1 = axis[0]
    ax2 = axis[1]

    p1, = ax1.plot(df.index, df['capital_vix_control'], color='r', linewidth=2, alpha=0.8)
    p2, = ax1.plot(df.index, df['capital_vol_control'], color='steelblue', linewidth=2)
    p3, = ax1.plot(df.index, df['capital_momentum'], color='orange', linewidth=2)
    p4, = ax1.plot(df.index, df['capital_risk_budget'], color='gray', linewidth=2, alpha=0.8)

    title1 = 'Cumulative Return from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax1.set_title(title1)
    ax1.legend([p1, p2, p3, p4], ['vix_control', 'sigma_control', strategy, 'risk_adjust'])
    ax1.grid(True)
    ax1.set_xlim(df.index[0], df.index[-1])

    f1 = ax2.fill_between(df.index, df['MaxDD_vix_control'], 0, facecolor='r', alpha=0.3)
    f2 = ax2.fill_between(df.index, df['MaxDD_vol_control'], 0, facecolor='steelblue', alpha=0.3)
    f3 = ax2.fill_between(df.index, df['MaxDD_momentum'], 0, facecolor='orange', alpha=0.3)
    f4 = ax2.fill_between(df.index, df['MaxDD_risk_budget'], 0, facecolor='gray', alpha=0.3)
    title2 = 'Drawdown from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax2.legend([f1, f2, f3, f4], ['vix_control', 'sigma_control', strategy, 'risk_adjust'], loc=3)
    ax2.set_title(title2)
    ax2.grid(True)
    ax2.set_xlim(df.index[0], df.index[-1])
    if isshow == True:
        plt.show()


def draw_plot1(df, strategy='momentum_strategy', isshow=True):
    """
    绘制回测曲线图
    strategy:{'vix_strategy', 'momentum_strategy'}
    """
    fig, axis = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(15, 8)
    ax1 = axis[0]
    ax2 = axis[1]

    p1, = ax1.plot(df.index, df['capital_risk_budget'], color='gray', linewidth=2, alpha=0.8)
    p2, = ax1.plot(df.index, df['capital_momentum'], color='r', linewidth=2)
    p3, = ax1.plot(df.index, df['capital_vol_control'], color='steelblue', linewidth=2)

    title1 = 'Cumulative Return from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax1.set_title(title1)
    ax1.legend([p1, p2, p3], ['risk_parity', strategy, 'volatility_control'])
    ax1.grid(True)
    ax1.set_xlim(df.index[0], df.index[-1])

    f1 = ax2.fill_between(df.index, df['MaxDD_risk_budget'], 0, facecolor='gray', alpha=0.3)
    f2 = ax2.fill_between(df.index, df['MaxDD_momentum'], 0, facecolor='r', alpha=0.3)
    f3 = ax2.fill_between(df.index, df['MaxDD_vol_control'], 0, facecolor='steelblue', alpha=0.3)

    title2 = 'Drawdown from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax2.legend([f1, f2, f3], ['risk_parity', strategy, 'volatility_control'], loc=3)
    ax2.set_title(title2)
    ax2.grid(True)
    ax2.set_xlim(df.index[0], df.index[-1])
    if isshow == True:
        plt.show()


def draw_plot2(df, isshow=True):
    """
    绘制回测曲线图
    strategy:{'vix_strategy', 'momentum_strategy'}
    """
    fig, axis = plt.subplots(2, 1, sharex=True)
    # fig.set_size_inches(15, 8)
    ax1 = axis[0]
    ax2 = axis[1]
    
    p1, = ax1.plot(df.index, df['风险资产'], color='steelblue', linewidth=2)
    p2, = ax1.plot(df.index, df['防御资产'], color='r', linewidth=2, alpha=0.8)

    title1 = 'Cumulative Return from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    # ax1.set_title(title1)
    ax1.legend([p1, p2], ['风险资产', '防御资产'])
    ax1.grid(True)
    ax1.set_xlim(df.index[0], df.index[-1])

    f1 = ax2.fill_between(df.index, df['MaxDD_rf'], 0, facecolor='r', alpha=0.3)
    f2 = ax2.fill_between(df.index, df['MaxDD_risky'], 0, facecolor='steelblue', alpha=0.3)
    
    title2 = 'Drawdown from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax2.legend([f1, f2], ['风险资产', '防御资产'], loc=3)
    # ax2.set_title(title2)
    ax2.grid(True)
    ax2.set_xlim(df.index[0], df.index[-1])
    if isshow == True:
        plt.show()
        
        
if __name__ == '__main__':
    equity = ['沪深300', 'SPX', '日经225', '德国DAX', '英国富时100', '美元指数', 'BCOMEN', 'BCOMIN']
    return_df = pd.read_excel(r'data/数据汇总1.0.xlsx', index_col=0)
    return_df = return_df[:'2019']
    vix_df = pd.read_csv(r'data/VIX_SHORT_INDEX.csv', index_col=1)
    vix_df.index = pd.to_datetime(vix_df.index)
    vix_df['vix_return_rate'] = vix_df['net_value'].pct_change().fillna(0)

    # 资产配对
    correlation = return_df["2015"].corr()
    match_list, unmatch_list = best_match(correlation, 1)

    # 风险调整确定初始权重
    weight_df, weight_rb = initial_pos(return_df, method=1)

    # 动量效应调整权重
    weight_m = strategy_select(return_df, weight_df, match_list, unmatch_list, method=0)

    # 获取风险调整后、动量调整后的收益率序列及动量调整后的杠杆率
    index_df, leverage_df = back_test(return_df, weight_m, weight_rb)

    # 波动率控制
    index_df, leverage_df = volatility_adjust(index_df, return_df, leverage_df, 0.03)

    # VIX调整
    index_df, leverage_df = vix_adjust(index_df, vix_df, leverage_df, equity) 
    weight_per_month = leverage_df.resample('M').mean()
    weight_per_month.to_excel('weight_per_month.xlsx')
    
    # 计算风险资产和防御资产月度收益率
    cal_risk_nonrisky_return(return_df, leverage_df, equity)

    # 回测净值曲线、回撤率
    index_df = cal_index_param_series(index_df)

    # 计算指标
    backtest_perform(index_df, 'VIX')

    # 绘图
    draw_plot1(index_df,  strategy='vix_strategy', isshow=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
