import numpy as np
import pandas as pd
from datetime import *
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


class Para:
    month1 = 32
    month2 = 24
    trd_cost1 = 0.006
    trd_cost2 = 0.005
    chg1 = 0.02
    chg2 = 0.04


para = Para()

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.simplefilter(action="ignore", category=RuntimeWarning)
os.chdir('C:/Users/ryuan/Desktop/')
retdata = pd.read_excel('指数日涨幅.xlsx', index_col='时间')

retdata.index = [int(datetime.strptime(i, "%Y-%m-%d").strftime('%Y%m%d')) for i in retdata.index]

# 每季度的第一个交易日
tmp = np.unique(np.floor(retdata.index / 100))
first_day = list()
for i in range(len(tmp)):
    first_day.append(min(retdata.index[np.floor(retdata.index / 100) == tmp[i]]))
last_day = list()
for i in range(len(tmp)):
    last_day.append(max(retdata.index[np.floor(retdata.index / 100) == tmp[i]]))

data_y0 = pd.DataFrame({'中证500': [0] * len(first_day), '沪深300': [0] * len(first_day)}, index=last_day)
for i in range(len(first_day)):
    y_tmp = retdata.loc[first_day[i]:last_day[i]]
    data_y0.loc[last_day[i], '中证500'] = (y_tmp['中证500'] / 100 + 1).cumprod().iloc[-1] - 1
    data_y0.loc[last_day[i], '沪深300'] = (y_tmp['沪深300'] / 100 + 1).cumprod().iloc[-1] - 1

label_y0 = pd.DataFrame({'label': [np.nan] * len(first_day)}, index=first_day)
for i in range(len(first_day)):
    y_tmp = data_y0.loc[last_day[i], '沪深300'] - data_y0.loc[last_day[i], '中证500']
    if y_tmp > para.chg1:
        label_y0.loc[first_day[i], 'label'] = 3
    elif y_tmp > 0:
        label_y0.loc[first_day[i], 'label'] = 2
    elif y_tmp < -para.chg1:
        label_y0.loc[first_day[i], 'label'] = 0
    else:
        label_y0.loc[first_day[i], 'label'] = 1
data_x0 = pd.read_excel('自变量2.xlsx', index_col='时间', sheetname='轮动1')
data_x0.index = [int(i.strftime('%Y%m%d')) for i in data_x0.index]
colnamestr = ['ER', 'M0', 'M2', 'CPI', 'PE', 'PB', 'PR', 'VOL', 'R1', 'V1', 'V3']
data_x0 = data_x0[colnamestr]

label_y1 = pd.DataFrame({'label': [np.nan] * len(first_day)}, index=first_day)
for i in range(len(first_day)):
    y_tmp = data_y0.loc[last_day[i], '沪深300'] - data_y0.loc[last_day[i], '中证500']
    if y_tmp > para.chg2:
        label_y1.loc[first_day[i], 'label'] = 3
    elif y_tmp > 0:
        label_y1.loc[first_day[i], 'label'] = 2
    elif y_tmp < -para.chg2:
        label_y1.loc[first_day[i], 'label'] = 0
    else:
        label_y1.loc[first_day[i], 'label'] = 1
data_x1 = pd.read_excel('自变量2.xlsx', index_col='时间', sheetname='择时')
data_x1.index = [int(i.strftime('%Y%m%d')) for i in data_x1.index]
colnamestr = ['GDP', 'CPI', 'PPI', 'CSM', 'INV', 'BOM', 'M1', 'M2', 'YTM10Y', 'YTM3m', 'pe300', 'pb300']
data_x1 = data_x1[colnamestr]

label_predict = pd.DataFrame(
    {'label': [0] * (len(data_x0) - para.month1), 'label1': [0] * (len(data_x0) - para.month1), 'sml': [0] * (len(data_x0) - para.month1)},
    index=label_y0.index[para.month1:])
strategy = pd.DataFrame({'return': [0] * len(retdata.loc[first_day[para.month1]:])}, index=retdata.loc[first_day[para.month1]:].index)
for d in range(len(data_x0) - para.month1):
    train_x = np.array(data_x0.iloc[d:(d + para.month1)])
    train_x[np.isnan(train_x)] = 0.
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    train_x = pd.DataFrame(train_x, index=first_day[d:(d + para.month1)])
    train_y = label_y0.iloc[d:(d + para.month1), 0]
    idx = train_y[np.isnan(train_y)].index
    train_x = train_x.drop(idx)
    train_y = train_y.drop(idx)
    classifier0 = LogisticRegression()  # 使用类，参数全是默认的
    classifier0.fit(train_x, train_y)  # 训练数据来学习，不需要返回值
    test_x = np.array(data_x0.iloc[d + para.month1])
    test_x[np.isnan(test_x)] = 0.
    predict_y = classifier0.predict(test_x.reshape(1, -1))[0]  # 测试数据，分类返回标记
    label_predict.loc[label_y0.index[d + para.month1], 'label'] = predict_y

    train_x = np.array(data_x1.iloc[(d + para.month1-para.month2):(d + para.month1)])
    train_x[np.isnan(train_x)] = 0.
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    train_x = pd.DataFrame(train_x, index=first_day[d:(d + para.month2)])
    train_y = label_y1.iloc[(d + para.month1-para.month2):(d + para.month1), 0]
    idx = train_y[np.isnan(train_y)].index
    train_x = train_x.drop(idx)
    train_y = train_y.drop(idx)
    classifier1 = LogisticRegression()  # 使用类，参数全是默认的
    classifier1.fit(train_x, train_y)  # 训练数据来学习，不需要返回值
    test_x = np.array(data_x1.iloc[d + para.month1])
    test_x[np.isnan(test_x)] = 0.
    predict_y1 = classifier1.predict(test_x.reshape(1, -1))[0]  # 测试数据，分类返回标记
    label_predict.loc[label_y0.index[d + para.month1], 'label1'] = predict_y1

    if predict_y <= 1:
        ret = retdata.loc[first_day[d + para.month1]:last_day[d + para.month1], '中证500'] / 100  # 小盘
        label_predict.loc[label_y0.index[d + para.month1], 'sml'] = 0
    else:
        ret = retdata.loc[first_day[d + para.month1]:last_day[d + para.month1], '沪深300'] / 100
        label_predict.loc[label_y0.index[d + para.month1], 'sml'] = 1
    if predict_y1 == 0:
        ret = 0.03 / 200
        label_predict.loc[label_y0.index[d + para.month1], 'sml'] = 2
    strategy.loc[first_day[d + para.month1]:last_day[d + para.month1], 'return'] = ret

strategy['nav'] = 0
ret = strategy.loc[first_day[para.month1]:last_day[para.month1], 'return']
strategy.loc[first_day[para.month1]:last_day[para.month1], 'nav'] = (1 + ret).cumprod() / (1 + para.trd_cost1)
for d in range(1, len(label_predict)):
    ret = strategy.loc[first_day[d + para.month1]:last_day[d + para.month1], 'return']
    if label_predict['sml'].iloc[d - 1] == label_predict['sml'].iloc[d]:
        strategy.loc[first_day[d + para.month1]:last_day[d + para.month1], 'nav'] = strategy.loc[last_day[d + para.month1 - 1], 'nav'] * (
            1 + ret).cumprod()
    else:
        strategy.loc[first_day[d + para.month1]:last_day[d + para.month1], 'nav'] = strategy.loc[last_day[d + para.month1 - 1], 'nav'] * (
            1 + ret).cumprod() / (1 + para.trd_cost1) * (1 - para.trd_cost2)

plot_data = (1 + retdata.loc[strategy.index] / 100).cumprod()
plot_data['strategy(cost)'] = strategy['nav']
plot_data['strategy'] = (1 + strategy['return']).cumprod()
plot_data = plot_data.iloc[-700:] / plot_data.iloc[-700]
day_list = plot_data.index
plot_data.index = range(len(plot_data))

plt.figure(1)
p1 = plt.subplot(2,1,1)
day_delta = (datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365
p1.plot(range(len(plot_data)), plot_data['沪深300'], "midnightblue",
         label='沪深300：年化收益率 ' + str(round(100 * (plot_data['沪深300'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
p1.plot(range(len(plot_data)), plot_data['中证500'], "steelblue",
         label='中证500：年化收益率 ' + str(round(100 * (plot_data['中证500'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
p1.plot(range(len(plot_data)), plot_data['strategy'], "mistyrose",
         label='策略组合：年化收益率 ' + str(round(100 * (plot_data['strategy'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
p1.plot(range(len(plot_data)), plot_data['strategy(cost)'], "red",
         label='策略组合(加交易成本)：年化收益率 ' + str(round(100 * (plot_data['strategy(cost)'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
plt.xticks(range(0, len(plot_data), 60), day_list[range(0, len(plot_data), 60)])
plt.legend(loc='upper left')
plt.title('logistic回归')
