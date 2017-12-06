import numpy as np
import pandas as pd
from datetime import *
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import talib
from sklearn import preprocessing


class Para:
    ma1 = 2
    ma2 = 10
    trix1 = 12
    trix2 = 20
    sdr = 2
    trd_period = 0
    trd_cost = 0.005
    vol = 0.01
    n_stock_select = 10


para = Para()

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.simplefilter(action="ignore", category=RuntimeWarning)
os.chdir('C:/Users/ryuan/Desktop/')
pricedata = pd.read_excel('index_price.xlsx', index_col=0, sheetname='Sheet1')

large_acc = pricedata['大盘指数(申万)']
small_acc = pricedata['小盘指数(申万)']
large_price = pricedata['大盘指数']
small_price = pricedata['小盘指数']

rsi = pd.DataFrame({'RSI': np.array(large_price) / np.array(small_price)}, index=pricedata.index)

TRIX = talib.TRIX(np.array(rsi['RSI']), timeperiod=para.trix1)
TRIX = np.array([np.nan] + list(TRIX[0:-1]))
MATRIX = talib.MA(TRIX, para.trix2)
MATRIX = np.array([np.nan] + list(MATRIX[0:-1]))

order_list_trix = np.zeros(len(rsi))  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
order_list_trix[TRIX > MATRIX] = 1
order_list_trix[TRIX < MATRIX] = 0

order_list_pr = np.zeros(len(rsi) - para.ma1)  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
order_list_pr[np.array(rsi['RSI'][0:-para.ma1]) < np.array(rsi['RSI'][para.ma1:])] = 1
order_list_pr[np.array(rsi['RSI'][0:-para.ma1]) > np.array(rsi['RSI'][para.ma1:])] = 0
order_list_pr = np.array([0] * para.ma1 + list(order_list_pr))

order_list = -1 * np.ones(len(rsi))
order_list_ma = order_list_pr
order_list[(order_list_ma == 1) & (order_list_trix == 1) & (np.array(talib.STDDEV(np.array(rsi['RSI']))) > para.vol)] = 1
order_list[(order_list_ma == 0) & (order_list_trix == 0) & (np.array(talib.STDDEV(np.array(rsi['RSI']))) > para.vol)] = 0
if order_list[0] == -1:
    order_list[0] = 0
for i in range(1, len(order_list)):
    if order_list[i] == -1:
        order_list[i] = order_list[i - 1]

funddata = pd.read_excel('fund.xlsx')  # 这部分是筛选后的基金，均为主动型、开放型、非分级基金
trade_days = np.array([int(i) for i in rsi.index.strftime('%Y%m%d')])
# 每季度的第一个交易日
tmp = np.unique(np.floor(trade_days / 100))
first_day = list()
for i in range(len(tmp)):
    first_day.append(min(trade_days[np.floor(trade_days / 100) == tmp[i]]))
last_day = list()
for i in range(len(tmp)):
    last_day.append(max(trade_days[np.floor(trade_days / 100) == tmp[i]]))
first_day = first_day[0::3]
last_day = last_day[2::3]
factor = ['abs_ret1', 'abs_ret3', 'beta', 'calmar1', 'calmar3', 'down1', 'down3', 'loss1', 'loss3', 'mdd1',
          'mdd3', 'sharp1', 'sharp3', 'stock', 'timing', 'vol1', 'vol3']
fundnav = pd.read_excel('基金周净值数据.xlsx', index_col=0)
index_tmp = [''] * len(fundnav)
for i in range(len(fundnav.index)):
    tmp = fundnav.index[i]
    index_tmp[i] = tmp[2:] + '.' + tmp[0:2]
fundnav.index = index_tmp
fundnav.columns = [int(datetime.strptime(i, "%Y-%m-%d").strftime('%Y%m%d')) for i in fundnav.columns]
benchnav = pd.read_excel('比较基准指数.xlsx', index_col=0)
benchnav.index = [int(datetime.strptime(i, "%Y-%m-%d").strftime('%Y%m%d')) for i in benchnav.index]
benchnav = benchnav / benchnav.iloc[0, :]

trd_signal = np.array(order_list[1:]) - np.array(order_list[0:-1])
trd_signal = np.array([0] + list(trd_signal))
trd_day = np.array(trade_days)[trd_signal != 0]
trd_day = np.array([trade_days[0]] + list(trd_day))
# weight1 = pd.read_excel('IC加权.xlsx', sheetname='Sheet1', index_col=0)
# weight2 = pd.read_excel('IC加权.xlsx', sheetname='Sheet2', index_col=0)
# weight1 = weight1.loc[factor, '偏股型']
# weight2 = weight2.loc[factor, '偏股型']
# weight = weight1 * weight2

ic_x = pd.DataFrame()
ic_y = pd.DataFrame()
type1 = ['大盘价值', '大盘成长', '大盘平衡','中盘价值', '中盘成长', '中盘平衡', '小盘价值', '小盘成长', '小盘平衡']
for i in range(list(trd_day).index(20101227)):
    d = trd_day[i]
    qd = np.array(first_day)[first_day <= d][-1]
    x = pd.DataFrame()
    y = pd.DataFrame()
    for k in type1:
        if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
            continue
        df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
        x = x.append(df_tmp_x[factor])
        if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'Y.csv'):
            continue
        df_tmp_y = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'Y.csv', index_col='code')
        y = y.append(df_tmp_y)
    ic_x = ic_x.append(x.reset_index())
    ic_y = ic_y.append(y.reset_index())

# for i in range(list(trd_day).index(20101227)):
#     d = trd_day[i]
#     qd = np.array(first_day)[first_day <= d][-1]
#     chg_date1 = np.array(fundnav.columns)[fundnav.columns >= d][0]
#     if i + 1 == len(trd_day):
#         chg_date2 = 20171117
#     else:
#         chg_date2 = np.array(fundnav.columns)[fundnav.columns >= trd_day[i + 1]][0]
#     if order_list[list(trade_days).index(d)] == 1:
#         type_tmp = ['大盘价值', '大盘成长', '大盘平衡']
#         x = pd.DataFrame()
#         for k in type_tmp:
#             if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
#                 continue
#             df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
#             x = x.append(df_tmp_x[factor])
#     else:
#         type_tmp = ['中盘价值', '中盘成长', '中盘平衡', '小盘价值', '小盘成长', '小盘平衡']
#         x = pd.DataFrame()
#         for k in type_tmp:
#             if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
#                 continue
#             df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
#             x = x.append(df_tmp_x[factor])
#     idx1 = x[np.sum(np.isinf(x), axis=1) > 0].index
#     idx2 = x[np.sum(np.isnan(x), axis=1) > 0].index
#     x = x.drop(idx1 | idx2)
#     all_y = fundnav.loc[x.index]
#     idx1 = all_y[np.sum(np.isinf(all_y), axis=1) > 0].index
#     idx2 = all_y[np.sum(np.isnan(all_y), axis=1) > 0].index
#     idx3 = all_y[np.sum(all_y.loc[:, chg_date1:chg_date2] == 0, axis=1) > 0].index
#     x = x.drop(idx1 | idx2 | idx3)
#     all_y = all_y.drop(idx1 | idx2 | idx3)
#     ic_x = ic_x.append(x.reset_index())
#     ic_y = ic_y.append((all_y[chg_date2] / all_y[chg_date1]).reset_index())

ic = pd.DataFrame(np.zeros(len(factor)))
ic.index = factor
for i in factor:
    ic.loc[i, 0] = np.corrcoef(np.array(ic_x[i]), np.array(ic_y['Y']))[0][1]
weight = ic[0]
# weight[abs(weight) < 0.3] = 0
# weight[abs(weight) > 0.3] = 1
# weight[abs(weight) == 1] = 1/sum(weight)

strategy = pd.DataFrame(
    {'return': [0] * (fundnav.shape[1] - 1), 'selection': [''] * (fundnav.shape[1] - 1), 'number': [0] * (fundnav.shape[1] - 1),
     'last5': [0] * (fundnav.shape[1] - 1), 'best5': [0] * (fundnav.shape[1] - 1), 'position': [0] * (fundnav.shape[1] - 1)}, index=fundnav.columns[0:-1])
chg_date1 = np.zeros(len(trd_day))
chg_date2 = np.zeros(len(trd_day))
for i in range(len(trd_day)):
    # ic = pd.DataFrame(np.zeros(len(factor)))
    # ic.index = factor
    # for k in factor:
    #     ic.loc[k, 0] = np.corrcoef(np.array(ic_x[k]), np.array(ic_y['Y']))[0][1]
    # weight = ic[0]
    d = trd_day[i]
    qd = np.array(first_day)[first_day <= d][-1]
    chg_date1[i] = np.array(fundnav.columns)[fundnav.columns >= d][0]
    if i + 1 == len(trd_day):
        chg_date2[i] = 20171117
    else:
        chg_date2[i] = np.array(fundnav.columns)[fundnav.columns >= trd_day[i + 1]][0]
    # x1 = pd.DataFrame()
    # y1 = pd.DataFrame()
    # for k in type1:
    #     if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
    #         continue
    #     df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
    #     x1 = x1.append(df_tmp_x[factor])
    #     if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'Y.csv'):
    #         continue
    #     df_tmp_y = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'Y.csv', index_col='code')
    #     y1 = y1.append(df_tmp_y)
    # ic_x = ic_x.append(x1.reset_index())
    # ic_y = ic_y.append(y1.reset_index())
    if order_list[list(trade_days).index(d)] == 1:
        type_tmp = ['大盘价值', '大盘成长', '大盘平衡']
        x = pd.DataFrame()
        for k in type_tmp:
            if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
                continue
            df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
            x = x.append(df_tmp_x[factor])
    else:
        type_tmp = ['中盘价值', '中盘成长', '中盘平衡', '小盘价值', '小盘成长', '小盘平衡']
        x = pd.DataFrame()
        for k in type_tmp:
            if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
                continue
            df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
            x = x.append(df_tmp_x[factor])
    idx1 = x[np.sum(np.isinf(x), axis=1) > 0].index
    idx2 = x[np.sum(np.isnan(x), axis=1) > 0].index
    x = x.drop(idx1 | idx2)
    all_y = fundnav.loc[x.index]
    idx1 = all_y[np.sum(np.isinf(all_y), axis=1) > 0].index
    idx2 = all_y[np.sum(np.isnan(all_y), axis=1) > 0].index
    idx3 = all_y[np.sum(all_y.loc[:, chg_date1[i]:chg_date2[i]] == 0, axis=1) > 0].index
    x = x.drop(idx1 | idx2 | idx3)
    all_y = all_y.drop(idx1 | idx2 | idx3)
    # scaler = preprocessing.StandardScaler().fit(x)
    # x1 = scaler.transform(x)
    score = pd.DataFrame({'score': np.dot(x, weight)}, index=x.index)
    y_score_curr_month = score.sort_values(by='score', ascending=False)
    if len(y_score_curr_month) < para.n_stock_select:
        n_stock_slc = len(y_score_curr_month)
    else:
        n_stock_slc = para.n_stock_select
    index_select = y_score_curr_month.iloc[0:n_stock_slc].index

    y = fundnav.loc[index_select]
    y1 = y.loc[:, chg_date1[i]:chg_date2[i]]
    y2 = np.mean(y1, axis=0)
    y3 = np.array(y2[1:]) / np.array(y2[0:-1]) - 1
    strategy.loc[y2.index[1:], 'return'] = y3
    strategy.loc[y2.index[1:], 'selection'] = '、'.join(index_select)
    strategy.loc[y2.index[1:], 'number'] = len(y)
    ly = fundnav.loc[y_score_curr_month.iloc[-n_stock_slc:].index]
    ly1 = ly.loc[:, chg_date1[i]:chg_date2[i]]
    ly2 = np.mean(ly1, axis=0)
    ly3 = np.array(ly2[1:]) / np.array(ly2[0:-1]) - 1
    strategy.loc[ly2.index[1:], 'last5'] = ly3
    ic_x = ic_x.append(x.reset_index())
    ic_y = ic_y.append((all_y[chg_date2[i]] / all_y[chg_date1[i]]).reset_index())
    best = (all_y[chg_date2[i]] / all_y[chg_date1[i]]).sort_values(ascending=False)
    best_index = best.iloc[0:n_stock_slc].index
    by = fundnav.loc[best_index]
    by1 = by.loc[:, chg_date1[i]:chg_date2[i]]
    by2 = np.mean(by1, axis=0)
    by3 = np.array(by2[1:]) / np.array(by2[0:-1]) - 1
    strategy.loc[y2.index[1:], 'best5'] = by3
    if order_list[list(trade_days).index(d)] == 1:
        strategy.loc[chg_date1[i], 'position'] = 1
    else:
        strategy.loc[chg_date1[i], 'position'] = -1

# strategy = pd.DataFrame(
#     {'return': [0] * fundnav.shape[1], 'selection': [''] * fundnav.shape[1], 'number': [0] * fundnav.shape[1], 'last5': [0] * fundnav.shape[1],
#      'best5': [0] * fundnav.shape[1]}, index=fundnav.columns)
# chg_date1 = np.zeros(len(trd_day))
# chg_date2 = np.zeros(len(trd_day))
# ic_x = pd.DataFrame()
# ic_y = pd.DataFrame()
# for i in range(len(trd_day)):
#     d = trd_day[i]
#     qd = np.array(first_day)[first_day <= d][-1]
#     if order_list[list(trade_days).index(d)] == 1:
#         type_tmp = ['大盘价值', '大盘成长', '大盘平衡']
#         x = pd.DataFrame()
#         for k in type_tmp:
#             if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
#                 continue
#             df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
#             x = x.append(df_tmp_x[factor])
#     else:
#         type_tmp = ['中盘价值', '中盘成长', '中盘平衡', '小盘价值', '小盘成长', '小盘平衡']
#         x = pd.DataFrame()
#         for k in type_tmp:
#             if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
#                 continue
#             df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
#             x = x.append(df_tmp_x[factor])
#     idx1 = x[np.sum(np.isinf(x), axis=1) > 0].index
#     idx2 = x[np.sum(np.isnan(x), axis=1) > 0].index
#     x = x.drop(idx1 | idx2)
#     all_y = fundnav.loc[x.index]
#     idx1 = all_y[np.sum(np.isinf(all_y), axis=1) > 0].index
#     idx2 = all_y[np.sum(np.isnan(all_y), axis=1) > 0].index
#     idx3 = all_y[np.sum(all_y.loc[:, chg_date1[i]:chg_date2[i]] == 0, axis=1) > 0].index
#     x = x.drop(idx1 | idx2 | idx3)
#     all_y = all_y.drop(idx1 | idx2 | idx3)
#     # scaler = preprocessing.StandardScaler().fit(x)
#     # x1 = scaler.transform(x)
#     chg_date1[i] = np.array(fundnav.columns)[fundnav.columns >= d][0]
#     if i + 1 == len(trd_day):
#         chg_date2[i] = 20171117
#     else:
#         chg_date2[i] = np.array(fundnav.columns)[fundnav.columns >= trd_day[i + 1]][0]
#     all_y2 = all_y.iloc[:,list(all_y.columns).index(chg_date1[i])]
#     if i == 0:
#         all_y1 = all_y.iloc[:, 0]
#     else:
#         all_y1 = all_y.iloc[:,list(all_y.columns).index(chg_date1[i])-4]
#     score = pd.DataFrame({'score': np.array( all_y2/all_y1) }, index=x.index)
#     # score = pd.DataFrame({'score': np.dot(x, weight)}, index=x.index)
#     y_score_curr_month = score.sort_values(by='score', ascending=False)
#     n_stock_select = 5  # int(np.ceil(len(x) / 10))
#     if len(y_score_curr_month) < n_stock_select:
#         n_stock_select = len(y_score_curr_month)
#     index_select = y_score_curr_month.iloc[0:n_stock_select].index
#
#     y = fundnav.loc[index_select]
#     y1 = y.loc[:, chg_date1[i]:chg_date2[i]]
#     y2 = np.mean(y1, axis=0)
#     y3 = np.array(y2[1:]) / np.array(y2[0:-1]) - 1
#     strategy.loc[y2.index[1:], 'return'] = y3
#     strategy.loc[y2.index[1:], 'selection'] = '、'.join(index_select)
#     strategy.loc[y2.index[1:], 'number'] = len(y)
#     ly = fundnav.loc[y_score_curr_month.iloc[-n_stock_select:].index]
#     ly1 = ly.loc[:, chg_date1[i]:chg_date2[i]]
#     ly2 = np.mean(ly1, axis=0)
#     ly3 = np.array(ly2[1:]) / np.array(ly2[0:-1]) - 1
#     strategy.loc[ly2.index[1:], 'last5'] = ly3
#     ic_x = ic_x.append(x.reset_index())
#     ic_y = ic_y.append((all_y[chg_date2[i]] / all_y[chg_date1[i]]).reset_index())
#     best = (all_y[chg_date2[i]] / all_y[chg_date1[i]]).sort_values(ascending=False)
#     best_index = best.iloc[0:n_stock_select].index
#     by = fundnav.loc[best_index]
#     by1 = by.loc[:, chg_date1[i]:chg_date2[i]]
#     by2 = np.mean(by1, axis=0)
#     by3 = np.array(by2[1:]) / np.array(by2[0:-1]) - 1
#     strategy.loc[y2.index[1:], 'best5'] = by3



portfolio = (1 + strategy['return']).cumprod()
last5funds = (1 + strategy['last5']).cumprod()
best5funds = (1 + strategy['best5']).cumprod()
portfolio1 = pd.DataFrame(portfolio.copy(), index=strategy.index)
for i in range(len(trd_day)):
    trd_day_b = trd_day[i]
    d1 = np.array(strategy.index)[strategy.index <= trd_day_b][-1]
    if i == len(trd_day) - 1:
        trd_day_n = 20091117
    else:
        trd_day_n = trd_day[i + 1]
    d2 = np.array(strategy.index)[strategy.index <= trd_day_n][-1]
    portfolio1.loc[d1:] = portfolio1.loc[d1:] / (1 + para.trd_cost)
    portfolio1.loc[d2:] = portfolio1.loc[d2:] * (1 - para.trd_cost)

last5funds1 = pd.DataFrame(last5funds.copy(), index=strategy.index)
for i in range(len(trd_day)):
    trd_day_b = trd_day[i]
    d1 = np.array(strategy.index)[strategy.index <= trd_day_b][-1]
    if i == len(trd_day) - 1:
        trd_day_n = 20091117
    else:
        trd_day_n = trd_day[i + 1]
    d2 = np.array(strategy.index)[strategy.index <= trd_day_n][-1]
    last5funds1.loc[d1:] = last5funds1.loc[d1:] / (1 + para.trd_cost)
    last5funds1.loc[d2:] = last5funds1.loc[d2:] * (1 - para.trd_cost)

best5funds1 = pd.DataFrame(best5funds.copy(), index=strategy.index)
for i in range(len(trd_day)):
    trd_day_b = trd_day[i]
    d1 = np.array(strategy.index)[strategy.index <= trd_day_b][-1]
    if i == len(trd_day) - 1:
        trd_day_n = 20091117
    else:
        trd_day_n = trd_day[i + 1]
    d2 = np.array(strategy.index)[strategy.index <= trd_day_n][-1]
    best5funds1.loc[d1:] = best5funds1.loc[d1:] / (1 + para.trd_cost)
    best5funds1.loc[d2:] = best5funds1.loc[d2:] * (1 - para.trd_cost)

strategy['portfolio'] = portfolio
strategy['portfolio1'] = portfolio1
strategy['last5funds'] = last5funds
strategy['last5funds1'] = last5funds1
strategy['best5funds'] = best5funds
strategy['best5funds1'] = best5funds1
strategy['hs300'] = benchnav['沪深300'][0:-1]
strategy['zz500'] = benchnav['中证500'][0:-1]
strategy['szzs'] = benchnav['上证指数'][0:-1]

pos = strategy.loc[20110107:, 'position']
strategy = strategy.loc[20110107:, 'portfolio':'szzs'] / np.array(strategy.loc[20110107, 'portfolio':'szzs'])

plt.figure(1, figsize=(16, 8))
p1 = plt.subplot(2, 1, 1)
daydelta = ((datetime.strptime(str(strategy.index[-1]), "%Y%m%d") - datetime.strptime(str(strategy.index[0]), "%Y%m%d")).days / 365)
p1.plot(range(len(strategy)), np.array(strategy['hs300']), "skyblue",
        label='沪深300：年化收益率 ' + str(round(100 * (strategy['hs300'].iloc[-1] ** (1 / daydelta) - 1),2)) + '%')

p1.plot(range(len(strategy)), np.array(strategy['zz500']), "deepskyblue",
        label='中证500：年化收益率 ' + str(round(100 * (strategy['zz500'].iloc[-1] ** (1 / daydelta) - 1),2)) + '%')

p1.plot(range(len(strategy)), np.array(strategy['szzs']), "midnightblue",
        label='上证指数：年化收益率 ' + str(round(100 * (strategy['szzs'].iloc[-1] ** (1 / daydelta) - 1),2)) + '%')
p1.plot(range(len(strategy)), np.array(strategy['portfolio']), "mistyrose",
        label='策略组合：年化收益率 ' + str(round(100 * (strategy['portfolio'].iloc[-1] ** (1 / daydelta) - 1),2)) + '%')
p1.plot(range(len(strategy)), np.array(strategy['portfolio1']), "red", linewidth=2,
        label='策略组合(加交易成本)：年化收益率 ' + str(round(100 * (strategy['portfolio1'].iloc[-1] ** (1 / daydelta) - 1),2)) + '%')
plt.xticks(range(0, len(strategy), 50), np.array([str(i) for i in strategy.index])[range(0, len(strategy), 50)])
plt.legend(loc='upper left')

p2 = plt.subplot(4, 1, 3)
pricerate = rsi.copy()
pricerate.index = [int(i.strftime('%Y%m%d')) for i in rsi.index]
p2.plot(range(len(pricerate.loc[20110107:20171117])), pricerate.loc[20110107:20171117], "grey", label='大小盘比价')
trd_sgn1 = pd.DataFrame([None] * len(pricerate.loc[20110107:20171117]), index=pricerate.loc[20110107:20171117].index)
trd_sgn2 = pd.DataFrame([None] * len(pricerate.loc[20110107:20171117]), index=pricerate.loc[20110107:20171117].index)
for i in range(len(pos)):
    if pos.iloc[i] == 1:
        trd_sgn1.loc[pos.index[i]] = float(pricerate.loc[pos.index[i]])
    elif pos.iloc[i] == -1:
        trd_sgn2.loc[pos.index[i]] = float(pricerate.loc[pos.index[i]])
p2.plot(range(len(trd_sgn1)), np.array(trd_sgn1), "ro")
p2.plot(range(len(trd_sgn2)), np.array(trd_sgn2), "go")
plt.xticks(range(0, len(trd_sgn1), 250), np.array([str(i) for i in trd_sgn1.index])[range(0, len(trd_sgn1), 250)])
plt.legend(loc='upper left')

large_position = order_list[1:]
small_position = 1 - order_list[1:]
large_ret = np.array(large_acc[1:]) / np.array(large_acc[0:-1]) - 1
small_ret = np.array(small_acc[1:]) / np.array(small_acc[0:-1]) - 1
ret = large_position * large_ret + small_position * small_ret
nav = (ret + 1).cumprod()
nav = np.array([1] + list(nav))

trd_day = np.array(list(range(len(rsi))))[trd_signal != 0]
nav1 = nav.copy()
for i in range(len(trd_day)):
    if i == 0:
        trd_day_b = 0
    else:
        trd_day_b = trd_day[i - 1]
    trd_day_n = trd_day[i]
    nav1[(trd_day_b + 1):] = nav1[(trd_day_b + 1):] * (1 - para.trd_cost)
    nav1[trd_day_n:] = nav1[trd_day_n:] * (1 - para.trd_cost)
navdata = pd.DataFrame({'nav':nav,'nav1':nav1},index=pricedata.index)
navdata = navdata.loc['2011-01-07':'2017-11-17']

p3 = plt.subplot(4, 1, 4)
p3.plot(range(len(navdata)), large_acc.loc['2011-01-07':'2017-11-17']/float(large_acc.loc['2011-01-07']), "skyblue",label='大盘指数')
p3.plot(range(len(navdata)), small_acc.loc['2011-01-07':'2017-11-17']/float(small_acc.loc['2011-01-07']), "deepskyblue",label='小盘指数')
p3.plot(range(len(navdata)), pricedata.loc['2011-01-07':'2017-11-17','沪深300净值']/pricedata.loc['2011-01-07','沪深300净值'], "midnightblue",label='沪深300')
p3.plot(range(len(navdata)), pricedata.loc['2011-01-07':'2017-11-17','上证净值']/pricedata.loc['2011-01-07','上证净值'], "steelblue",label='上证指数')
p3.plot(range(len(navdata)), navdata['nav']/navdata['nav'][0], "mistyrose", label='策略组合：年化收益率 '+str(round(100 * ((navdata['nav']/navdata['nav'][0])[-1] ** (1 / daydelta) - 1), 2)) + '%')
p3.plot(range(len(navdata)), navdata['nav1']/navdata['nav1'][0], "red", label='策略组合(加交易成本)：年化收益率 '+str(round(100 * ((navdata['nav1']/navdata['nav1'][0])[-1] ** (1 / daydelta) - 1), 2)) + '%')
# p1.plot(range(len(nav)), np.array(talib.STDDEV(np.array(rsi['RSI'])))*1000, "red", label='波动')
plt.xticks(range(0, len(navdata), 180), navdata.index.strftime('%Y-%m')[range(0, len(navdata), 180)])
plt.legend(loc='upper left')

# strategy = pd.DataFrame({'return': [0] * fundnav.shape[1], 'selection': [''] * fundnav.shape[1], 'number': [0] * fundnav.shape[1]}, index=fundnav.columns)
# last5 = pd.DataFrame({'return': [0] * fundnav.shape[1]}, index=fundnav.columns)
# best5 = pd.DataFrame({'return': [0] * fundnav.shape[1]}, index=fundnav.columns)
# chg_date1 = np.zeros(len(first_day))
# chg_date2 = np.zeros(len(first_day))
# ic_x = pd.DataFrame()
# ic_y = pd.DataFrame()
# weight1 = pd.read_excel('IC加权.xlsx', sheetname='Sheet1', index_col=0)
# weight2 = pd.read_excel('IC加权.xlsx', sheetname='Sheet2', index_col=0)
# weight1 = weight1.loc[factor, '偏股型']
# weight2 = weight2.loc[factor, '偏股型']
# weight = weight1 * weight2
# for i in range(len(first_day)):
#     d = first_day[i]
#     qd = np.array(first_day)[first_day <= d][-1]
#     if order_list[list(trade_days).index(d)] == 1:
#         type_tmp = ['大盘价值', '大盘成长', '大盘平衡']
#         x = pd.DataFrame()
#         for k in type_tmp:
#             if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
#                 continue
#             df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
#             x = x.append(df_tmp_x[factor])
#     else:
#         type_tmp = ['中盘价值', '中盘成长', '中盘平衡', '小盘价值', '小盘成长', '小盘平衡']
#         x = pd.DataFrame()
#         for k in type_tmp:
#             if not os.path.exists('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv'):
#                 continue
#             df_tmp_x = pd.read_csv('Data2/偏股型/' + k + '/' + str(qd) + 'X.csv', index_col='code')
#             x = x.append(df_tmp_x[factor])
#     idx1 = x[np.sum(np.isinf(x), axis=1) > 0].index
#     idx2 = x[np.sum(np.isnan(x), axis=1) > 0].index
#     x = x.drop(idx1 | idx2)
#     all_y = fundnav.loc[x.index]
#     idx1 = all_y[np.sum(np.isinf(all_y), axis=1) > 0].index
#     idx2 = all_y[np.sum(np.isnan(all_y), axis=1) > 0].index
#     idx3 = all_y[np.sum(all_y.loc[:, chg_date1[i]:chg_date2[i]] == 0, axis=1) > 0].index
#     x = x.drop(idx1 | idx2 | idx3)
#     all_y = all_y.drop(idx1 | idx2 | idx3)
#     # scaler = preprocessing.StandardScaler().fit(x)
#     # x1 = scaler.transform(x)
#     score = pd.DataFrame({'score': np.dot(x, weight)}, index=x.index)
#     y_score_curr_month = score.sort_values(by='score', ascending=False)
#     n_stock_select = 5  # int(np.ceil(len(x) / 10))
#     if len(y_score_curr_month) < n_stock_select:
#         n_stock_select = len(y_score_curr_month)
#     index_select = y_score_curr_month.iloc[0:n_stock_select].index
#     chg_date1[i] = np.array(fundnav.columns)[fundnav.columns >= d][0]
#     if i + 1 == len(first_day):
#         chg_date2[i] = 20171117
#     else:
#         chg_date2[i] = np.array(fundnav.columns)[fundnav.columns >= first_day[i + 1]][0]
#     y = fundnav.loc[index_select]
#     y1 = y.loc[:, chg_date1[i]:chg_date2[i]]
#     y2 = np.mean(y1, axis=0)
#     y3 = np.array(y2[1:]) / np.array(y2[0:-1]) - 1
#     strategy.loc[y2.index[1:], 'return'] = y3
#     strategy.loc[y2.index[1:], 'selection'] = '、'.join(index_select)
#     strategy.loc[y2.index[1:], 'number'] = len(y)
#     ly = fundnav.loc[y_score_curr_month.iloc[-n_stock_select:].index]
#     ly1 = ly.loc[:, chg_date1[i]:chg_date2[i]]
#     ly2 = np.mean(ly1, axis=0)
#     ly3 = np.array(ly2[1:]) / np.array(ly2[0:-1]) - 1
#     strategy.loc[ly2.index[1:], 'last5'] = ly3
#     ic_x = ic_x.append(x.reset_index())
#     ic_y = ic_y.append((all_y[chg_date2[i]] / all_y[chg_date1[i]]).reset_index())
#     best = (all_y[chg_date2[i]] / all_y[chg_date1[i]]).sort_values(ascending=False)
#     best_index = best.iloc[0:n_stock_select].index
#     by = fundnav.loc[best_index]
#     by1 = by.loc[:, chg_date1[i]:chg_date2[i]]
#     by2 = np.mean(by1, axis=0)
#     by3 = np.array(by2[1:]) / np.array(by2[0:-1]) - 1
#     strategy.loc[y2.index[1:], 'best5'] = by3
