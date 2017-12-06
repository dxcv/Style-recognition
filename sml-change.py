import numpy as np
import pandas as pd
from datetime import *
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import talib


class Para:
    ma1 = 2
    ma2 = 10
    trix1 = 12
    trix2 = 20
    sdr = 2
    trd_period = 0
    trd_cost = 0.005
    vol = 0.01


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

# rsi = pd.DataFrame({'RSI': np.log(large_acc) - np.log(small_acc)}, index=pricedata.index)
# rsi.to_excel('large_small_rsi.xlsx')
rsi = pd.DataFrame({'RSI': np.array(large_price) / np.array(small_price)}, index=pricedata.index)

# cp_mean1 = np.zeros(len(rsi))
# cp_mean2 = np.zeros(len(rsi))
# for di in range(para.ma2, len(rsi)):
#     cp_list1 = rsi['RSI'].iloc[(di - para.ma1):di]  # 使用前一天的数据，避免未来函数
#     cp_array1 = np.array(cp_list1)
#     cp_mean1[di] = cp_array1.mean()
#     cp_list2 = rsi['RSI'].iloc[(di - para.ma2):di]
#     cp_array2 = np.array(cp_list2)
#     cp_mean2[di] = cp_array2.mean()
cp_mean1 = talib.MA(np.array(rsi['RSI']), para.ma1)
cp_mean1 = np.array([np.nan] + list(cp_mean1[0:-1]))
cp_mean2 = talib.MA(np.array(rsi['RSI']), para.ma2)
cp_mean2 = np.array([np.nan] + list(cp_mean2[0:-1]))

TRIX = talib.TRIX(np.array(rsi['RSI']), timeperiod=para.trix1)
TRIX = np.array([np.nan] + list(TRIX[0:-1]))
MATRIX = talib.MA(TRIX, para.trix2)
MATRIX = np.array([np.nan] + list(MATRIX[0:-1]))

# order_list_ma = np.zeros(len(rsi))  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
# order_list_ma[cp_mean2 > cp_mean1] = 0
# order_list_ma[cp_mean2 < cp_mean1] = 1

# order_list_trixma = np.zeros(len(rsi))  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
# order_list_trixma[(TRIX > MATRIX)&(abs(TRIX-MATRIX)>0.01)] = 1
# order_list_trixma[(TRIX < MATRIX)&(abs(MATRIX-TRIX)>0.01)] = 0
order_list_trix = np.zeros(len(rsi))  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
order_list_trix[TRIX > MATRIX] = 1
order_list_trix[TRIX < MATRIX] = 0

order_list_pr = np.zeros(len(rsi) - para.ma1)  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
order_list_pr[np.array(rsi['RSI'][0:-para.ma1]) < np.array(rsi['RSI'][para.ma1:])] = 1
order_list_pr[np.array(rsi['RSI'][0:-para.ma1]) > np.array(rsi['RSI'][para.ma1:])] = 0
order_list_pr = np.array([0] * para.ma1 + list(order_list_pr))

# order_list = order_list_trix
order_list = -1 * np.ones(len(rsi))
order_list_ma = order_list_pr
order_list[(order_list_ma == 1) & (order_list_trix == 1) & (np.array(talib.STDDEV(np.array(rsi['RSI']))) > para.vol)] = 1
order_list[(order_list_ma == 0) & (order_list_trix == 0) & (np.array(talib.STDDEV(np.array(rsi['RSI']))) > para.vol)] = 0
if order_list[0] == -1:
    order_list[0] = 0
for i in range(1, len(order_list)):
    if order_list[i] == -1:
        order_list[i] = order_list[i - 1]

# 过滤交易信号
order_list2 = order_list.copy()
trd_signal = np.array(order_list2[1:]) - np.array(order_list2[0:-1])
trd_signal = np.array([0] + list(trd_signal))
for i in range(para.trd_period, len(rsi)):
    if (sum(abs(trd_signal[(i - para.trd_period):i])) > 0) & (abs(trd_signal[i]) > 0):
        order_list2[i] = order_list2[i - 1]
        trd_signal = np.array(order_list2[1:]) - np.array(order_list2[0:-1])
        trd_signal = np.array([0] + list(trd_signal))
for i in range(1, len(rsi) - para.trd_period):
    if order_list2[i] - order_list2[i - 1] != 0:
        order_list2[i:(i + para.trd_period)] = np.ones(para.trd_period) * order_list2[i + 1]
trd_signal = np.array(order_list2[1:]) - np.array(order_list2[0:-1])
trd_signal = np.array([0] + list(trd_signal))

large_position = order_list2[1:]
small_position = 1 - order_list2[1:]
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
plt.figure(1, figsize=(8, 6))
p1 = plt.subplot(2, 1, 1)
p1.plot(range(len(nav)), large_acc, "skyblue",
        label='大盘指数：年化收益率 ' + str(round(100 * (large_acc[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), small_acc, "deepskyblue",
        label='小盘指数：年化收益率 ' + str(round(100 * (small_acc[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
# p1.plot(range(len(nav)), rsi['RSI']*10, "grey", label='大小盘比价')
p1.plot(range(len(nav)), pricedata['沪深300净值'], "midnightblue",
        label='沪深300：年化收益率 ' + str(round(100 * (pricedata['沪深300净值'][-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), pricedata['上证净值'], "steelblue",
        label='上证指数：年化收益率 ' + str(round(100 * (pricedata['上证净值'][-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), nav, "mistyrose", label='策略组合：年化收益率 '+str(round(100 * (nav[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), nav1, "red", label='策略组合(加交易成本)：年化收益率 '+str(round(100 * (nav1[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
# p1.plot(range(len(nav)), np.array(talib.STDDEV(np.array(rsi['RSI'])))*1000, "red", label='波动')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
# p1.plot(range(len(rsi['RSI'])), rsi['RSI'], "red")
# p1.plot(range(len(rsi['RSI'])), cp_mean1, "b")
# p1.plot(range(len(rsi['RSI'])), cp_mean2, "g")
p2 = plt.subplot(8, 1, 5)
# p2.plot(range(len(rsi['RSI'])), order_list, "y")
# p2.plot(range(len(rsi['RSI'])), order_list2, "r--")
r = np.array(nav[1:])/np.array(nav[0:-1])-1
best_r = np.zeros(len(r))
for i in range(len(rsi)-1):
    if large_price[i+1] / large_price[i] - small_price[i+1] / small_price[i] > 0:
        best_r = large_price[i+1] / large_price[i]-1
    else:
        best_r = small_price[i + 1] / small_price[i] - 1
day_win = sum(r>=best_r)/len(rsi)
win_rate = np.zeros(len(nav))  # 胜率
lose_rate = np.zeros(len(nav))  # 错误率
trd_day = np.array(list(range(len(rsi))))[trd_signal != 0]
for i in range(len(trd_day) + 1):
    if i == 0:
        trd_day_b = 0
    else:
        trd_day_b = trd_day[i - 1]
    if i == len(trd_day):
        trd_day_n = len(rsi) - 1
    else:
        trd_day_n = trd_day[i]
    if sum(order_list2[trd_day_b:trd_day_n]) > 0:
        if large_price[trd_day_n] / large_price[trd_day_b] - small_price[trd_day_n] / small_price[trd_day_b] > 0:
            win_rate[trd_day_b:trd_day_n] = 1
        else:
            lose_rate[trd_day_b:trd_day_n] = 1
    else:
        if large_price[trd_day_n] / large_price[trd_day_b] - small_price[trd_day_n] / small_price[trd_day_b] < 0:
            win_rate[trd_day_b:trd_day_n] = -1
        else:
            lose_rate[trd_day_b:trd_day_n] = -1
p2.bar(range(len(rsi['RSI'])), win_rate, color="darkred", alpha=0.5, width=1, label='区间正确率' + str(round(sum(abs(win_rate)) / len(win_rate) * 100, 2)) + '%')
p2.bar(range(len(rsi['RSI'])), lose_rate, color="darkgreen", alpha=0.5, width=1,
       label='交易次数:' + str(len(trd_day)) + '次, 平均交易周期:' + str(int(np.mean(trd_day[1:] - trd_day[0:-1]))) + '天')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
p3 = plt.subplot(8, 1, 6)
p3.plot(range(len(rsi['RSI'])), TRIX, "darkcyan", label='TRIX')
p3.plot(range(len(rsi['RSI'])), MATRIX, "chocolate", label='MATRIX')
plt.legend(loc='upper left')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
p4 = plt.subplot(8, 1, 7)
p4.plot(range(len(rsi['RSI'])), [np.nan] * para.ma1 + list(np.array(rsi['RSI'][0:-para.ma1])), "darkcyan", linewidth=0.5)
p4.plot(range(len(rsi['RSI'])), [np.nan] * para.ma1 + list(np.array(rsi['RSI'][para.ma1:])), "chocolate", linewidth=0.5, label='比价')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
p5 = plt.subplot(8, 1, 8)
p5.plot(range(len(rsi['RSI'])), np.array(talib.STDDEV(np.array(rsi['RSI']))), "grey", label='波动率')
p5.plot(range(len(rsi['RSI'])), [para.vol] * len(rsi['RSI']), "b--", linewidth=0.5)
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
##############################################

class Para:
    ma1 = 2
    ma2 = 10
    trix1 = 12
    trix2 = 20
    sdr = 2
    trd_period = 0
    trd_cost = 0.005
    vol = 0.005

para=Para()
pricedata2 = pd.read_excel('index_price.xlsx', index_col=0, sheetname='Sheet2')

large_acc = pricedata2['价值']
small_acc = pricedata2['成长']
large_price = pricedata2['全指价值']
small_price = pricedata2['全指成长']

rsi = pd.DataFrame({'RSI': np.array(large_price) / np.array(small_price)}, index=pricedata2.index)

TRIX = talib.TRIX(np.array(rsi['RSI']), timeperiod=para.trix1)
TRIX = np.array([np.nan] + list(TRIX[0:-1]))
MATRIX = talib.MA(TRIX, para.trix2)
MATRIX = np.array([np.nan] + list(MATRIX[0:-1]))

# order_list_ma = np.zeros(len(rsi))  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
# order_list_ma[cp_mean2 > cp_mean1] = 0
# order_list_ma[cp_mean2 < cp_mean1] = 1

# order_list_trixma = np.zeros(len(rsi))  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
# order_list_trixma[(TRIX > MATRIX)&(abs(TRIX-MATRIX)>0.01)] = 1
# order_list_trixma[(TRIX < MATRIX)&(abs(MATRIX-TRIX)>0.01)] = 0
order_list_trix = np.zeros(len(rsi))  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
order_list_trix[TRIX > MATRIX] = 1
order_list_trix[TRIX < MATRIX] = 0

order_list_pr = np.zeros(len(rsi) - para.ma1)  # 0为持有小盘(1->0买入小盘卖出大盘)，1为持有大盘(0->1买入大盘卖出小盘)，初始状态持有小盘
order_list_pr[np.array(rsi['RSI'][0:-para.ma1]) < np.array(rsi['RSI'][para.ma1:])] = 1
order_list_pr[np.array(rsi['RSI'][0:-para.ma1]) > np.array(rsi['RSI'][para.ma1:])] = 0
order_list_pr = np.array([0] * para.ma1 + list(order_list_pr))

# order_list = order_list_trix
order_list = -1 * np.ones(len(rsi))
order_list_ma = order_list_pr
order_list[(order_list_ma == 1) & (order_list_trix == 1) & (np.array(talib.STDDEV(np.array(rsi['RSI']))) > para.vol)] = 1
order_list[(order_list_ma == 0) & (order_list_trix == 0) & (np.array(talib.STDDEV(np.array(rsi['RSI']))) > para.vol)] = 0
if order_list[0] == -1:
    order_list[0] = 0
for i in range(1, len(order_list)):
    if order_list[i] == -1:
        order_list[i] = order_list[i - 1]

# 过滤交易信号
order_list2 = order_list.copy()
trd_signal = np.array(order_list2[1:]) - np.array(order_list2[0:-1])
trd_signal = np.array([0] + list(trd_signal))
for i in range(para.trd_period, len(rsi)):
    if (sum(abs(trd_signal[(i - para.trd_period):i])) > 0) & (abs(trd_signal[i]) > 0):
        order_list2[i] = order_list2[i - 1]
        trd_signal = np.array(order_list2[1:]) - np.array(order_list2[0:-1])
        trd_signal = np.array([0] + list(trd_signal))
for i in range(1, len(rsi) - para.trd_period):
    if order_list2[i] - order_list2[i - 1] != 0:
        order_list2[i:(i + para.trd_period)] = np.ones(para.trd_period) * order_list2[i + 1]
trd_signal = np.array(order_list2[1:]) - np.array(order_list2[0:-1])
trd_signal = np.array([0] + list(trd_signal))

large_position = order_list2[1:]
small_position = 1 - order_list2[1:]
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

plt.figure(2, figsize=(8, 6))
p1 = plt.subplot(2, 1, 1)
p1.plot(range(len(nav)), large_acc, "skyblue",
        label='全指价值：年化收益率 ' + str(round(100 * (large_acc[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), small_acc, "deepskyblue",
        label='全指成长：年化收益率 ' + str(round(100 * (small_acc[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
# p1.plot(range(len(nav)), rsi['RSI']*10, "grey", label='大小盘比价')
hs300 = pricedata.loc[rsi.index, '沪深300净值'] / float(pricedata.loc[rsi.index[0], '沪深300净值'])
szzs = pricedata.loc[rsi.index, '上证净值'] / float(pricedata.loc[rsi.index[0], '上证净值'])
p1.plot(range(len(nav)), hs300, "midnightblue",
        label='沪深300：年化收益率 ' + str(round(100 * (hs300[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), szzs, "steelblue",
        label='上证指数：年化收益率 ' + str(round(100 * (szzs[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), nav, "mistyrose", label='策略组合：年化收益率 '+str(round(100 * (nav[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
p1.plot(range(len(nav)), nav1, "red", label='策略组合(加交易成本)：年化收益率 '+str(round(100 * (nav1[-1] ** (1 / ((rsi.index[-1] - rsi.index[0]).days / 365)) - 1), 2)) + '%')
# p1.plot(range(len(nav)), np.array(talib.STDDEV(np.array(rsi['RSI'])))*1000, "red", label='波动')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
# p1.plot(range(len(rsi['RSI'])), rsi['RSI'], "red")
# p1.plot(range(len(rsi['RSI'])), cp_mean1, "b")
# p1.plot(range(len(rsi['RSI'])), cp_mean2, "g")
p2 = plt.subplot(8, 1, 5)
# p2.plot(range(len(rsi['RSI'])), order_list, "y")
# p2.plot(range(len(rsi['RSI'])), order_list2, "r--")
win_rate = np.zeros(len(nav))  # 胜率
lose_rate = np.zeros(len(nav))  # 错误率
trd_day = np.array(list(range(len(rsi))))[trd_signal != 0]
for i in range(len(trd_day) + 1):
    if i == 0:
        trd_day_b = 0
    else:
        trd_day_b = trd_day[i - 1]
    if i == len(trd_day):
        trd_day_n = len(rsi) - 1
    else:
        trd_day_n = trd_day[i]
    if sum(order_list2[trd_day_b:trd_day_n]) > 0:
        if large_price[trd_day_n] / large_price[trd_day_b] - small_price[trd_day_n] / small_price[trd_day_b] > 0:
            win_rate[trd_day_b:trd_day_n] = 1
        else:
            lose_rate[trd_day_b:trd_day_n] = 1
    else:
        if large_price[trd_day_n] / large_price[trd_day_b] - small_price[trd_day_n] / small_price[trd_day_b] < 0:
            win_rate[trd_day_b:trd_day_n] = -1
        else:
            lose_rate[trd_day_b:trd_day_n] = -1
p2.bar(range(len(rsi['RSI'])), win_rate, color="darkred", alpha=0.5, width=1, label='区间正确率' + str(round(sum(abs(win_rate)) / len(win_rate) * 100, 2)) + '%')
p2.bar(range(len(rsi['RSI'])), lose_rate, color="darkgreen", alpha=0.5, width=1,
       label='交易次数:' + str(len(trd_day)) + '次, 平均交易周期:' + str(int(np.mean(trd_day[1:] - trd_day[0:-1]))) + '天')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
p3 = plt.subplot(8, 1, 6)
p3.plot(range(len(rsi['RSI'])), TRIX, "darkcyan", label='TRIX')
p3.plot(range(len(rsi['RSI'])), MATRIX, "chocolate", label='MATRIX')
plt.legend(loc='upper left')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
p4 = plt.subplot(8, 1, 7)
p4.plot(range(len(rsi['RSI'])), [np.nan] * para.ma1 + list(np.array(rsi['RSI'][0:-para.ma1])), "darkcyan", linewidth=0.5)
p4.plot(range(len(rsi['RSI'])), [np.nan] * para.ma1 + list(np.array(rsi['RSI'][para.ma1:])), "chocolate", linewidth=0.5, label='比价')
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
p5 = plt.subplot(8, 1, 8)
p5.plot(range(len(rsi['RSI'])), np.array(talib.STDDEV(np.array(rsi['RSI']))), "grey", label='波动率')
p5.plot(range(len(rsi['RSI'])), [para.vol] * len(rsi['RSI']), "b--", linewidth=0.5)
plt.xticks(range(0, len(nav), 180), rsi.index.strftime('%Y-%m')[range(0, len(nav), 180)])
plt.legend(loc='upper left')
