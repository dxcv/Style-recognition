import numpy as np
import pandas as pd
from datetime import *
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


class Para:
    month = 30
    trd_cost1 = 0.006
    trd_cost2 = 0.005
    chg1 = 0.01
    chg2 = 0.02


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

estm = pd.DataFrame()
for m1 in range(12, 40, 6):
    for j1 in np.arange(0, 0.05, 0.01):
        para.chg1 = j1
        para.month = m1
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
        data_x0 = pd.read_excel('自变量2.xlsx', index_col='时间', sheetname='择时')
        data_x0.index = [int(i.strftime('%Y%m%d')) for i in data_x0.index]
        colnamestr = ['GDP', 'CPI', 'PPI', 'CSM', 'INV', 'BOM', 'M1', 'M2', 'YTM10Y', 'YTM3m', 'pe300', 'pb300','MACD', 'ZD', 'macd', 'dif', 'dea']
        data_x0 = data_x0[colnamestr]

        label_predict = pd.DataFrame(
            {'label': [0] * (len(data_y0) - para.month), 'label1': [0] * (len(data_y0) - para.month), 'sml': [0] * (len(data_y0) - para.month)},
            index=label_y0.index[para.month:])
        strategy = pd.DataFrame({'return': [0] * len(retdata.loc[first_day[para.month]:])}, index=retdata.loc[first_day[para.month]:].index)
        for d in range(len(data_x0) - para.month):
            train_x = np.array(data_x0.iloc[d:(d + para.month)])
            train_x[np.isnan(train_x)] = 0.
            # scaler = preprocessing.StandardScaler().fit(train_x)
            # train_x = scaler.transform(train_x)
            train_x = pd.DataFrame(train_x, index=first_day[d:(d + para.month)])
            train_y = label_y0.iloc[d:(d + para.month), 0]
            idx = train_y[np.isnan(train_y)].index
            train_x = train_x.drop(idx)
            train_y = train_y.drop(idx)
            classifier0 = LogisticRegression()  # 使用类，参数全是默认的
            classifier0.fit(train_x, train_y)  # 训练数据来学习，不需要返回值
            test_x = np.array(data_x0.iloc[d + para.month])
            test_x[np.isnan(test_x)] = 0.
            predict_y = classifier0.predict(test_x.reshape(1, -1))[0]  # 测试数据，分类返回标记
            label_predict.loc[label_y0.index[d + para.month], 'label'] = predict_y
            if predict_y <= 0:
                ret = 0.03/200
                label_predict.loc[label_y0.index[d + para.month], 'sml'] = 0
            else:
                ret = retdata.loc[first_day[d + para.month]:last_day[d + para.month], '沪深300'] / 100
                label_predict.loc[label_y0.index[d + para.month], 'sml'] = 1
            strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'return'] = ret

        strategy['nav'] = 0
        ret = strategy.loc[first_day[para.month]:last_day[para.month], 'return']
        strategy.loc[first_day[para.month]:last_day[para.month], 'nav'] = (1 + ret).cumprod() / (1 + para.trd_cost1)
        for d in range(1, len(label_predict)):
            ret = strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'return']
            if label_predict['sml'].iloc[d - 1] == label_predict['sml'].iloc[d]:
                strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'nav'] = strategy.loc[last_day[d + para.month - 1], 'nav'] * (
                    1 + ret).cumprod()
            else:
                strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'nav'] = strategy.loc[last_day[d + para.month - 1], 'nav'] * (
                    1 + ret).cumprod() / (1 + para.trd_cost1) * (1 - para.trd_cost2)
        day_list = strategy.index
        estm.loc[str(m1) + str(j1), 'ret1'] = ((1 + strategy['return']).cumprod()).iloc[-1] ** (
            1 / ((datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365)) - 1
        strategy = strategy.iloc[-700:] / strategy.iloc[-700]
        day_list = strategy.index
        estm.loc[str(m1) + str(j1), 'ret'] = strategy['nav'].iloc[-1] ** (
            1 / ((datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365)) - 1
        estm.loc[str(m1) + str(j1), 'chg1'] = j1
        estm.loc[str(m1) + str(j1), 'mon'] = m1
        label_predict = label_predict.loc[20150116:]
        estm.loc[str(m1) + str(j1), 'p'] = (np.sum((label_predict['sml'] == 0) & (label_y0.loc[label_predict.index, 'label'] <= 1)) + np.sum(
            (label_predict['sml'] >= 1) & (label_y0.loc[label_predict.index, 'label'] > 1))) / len(label_predict)
estm[estm['ret']==max(estm['ret'])]




estm = pd.DataFrame()
# for m1 in range(20, 40, 4):
#     for j1 in np.arange(0, 0.05, 0.01):
for j in reversed(range(16,23,1)):
    # para.chg1 = j1
    # para.month = m1
    label_y0 = pd.DataFrame({'label': [np.nan] * len(first_day)}, index=first_day)
    for i in range(len(first_day)):
        y_tmp = max(data_y0.loc[last_day[i], '沪深300'], data_y0.loc[last_day[i], '中证500'])
        if y_tmp > para.chg2:
            label_y0.loc[first_day[i], 'label'] = 2
        elif y_tmp > -para.chg1:
            label_y0.loc[first_day[i], 'label'] = 1
        else:
            label_y0.loc[first_day[i], 'label'] = 0
    data_x01 = pd.read_excel('自变量2.xlsx', index_col='时间', sheetname='择时')
    data_x01.index = [int(i.strftime('%Y%m%d')) for i in data_x01.index]
    # clnm = ['ER', 'M0', 'M2', 'CPI', 'PR', 'INR', 'VOL', 'R3', 'V1', 'DEA', 'MACD']
    clnm = data_x01.columns
    for i in list(itertools.combinations(clnm, j)):
        print(i)
        data_x0 = data_x01[list(i)]
        label_predict = pd.DataFrame(
            {'label': [0] * (len(data_y0) - para.month), 'label1': [0] * (len(data_y0) - para.month), 'sml': [0] * (len(data_y0) - para.month)},
            index=label_y0.index[para.month:])
        strategy = pd.DataFrame({'return': [0] * len(retdata.loc[first_day[para.month]:])}, index=retdata.loc[first_day[para.month]:].index)
        for d in range(len(data_x0) - para.month):
            train_x = np.array(data_x0.iloc[d:(d + para.month)])
            train_x[np.isnan(train_x)] = 0.
            # scaler = preprocessing.StandardScaler().fit(train_x)
            # train_x = scaler.transform(train_x)
            train_x = pd.DataFrame(train_x, index=first_day[d:(d + para.month)])
            train_y = label_y0.iloc[d:(d + para.month), 0]
            idx = train_y[np.isnan(train_y)].index
            train_x = train_x.drop(idx)
            train_y = train_y.drop(idx)
            classifier0 = LogisticRegression()  # 使用类，参数全是默认的
            classifier0.fit(train_x, train_y)  # 训练数据来学习，不需要返回值
            test_x = np.array(data_x0.iloc[d + para.month])
            test_x[np.isnan(test_x)] = 0.
            predict_y = classifier0.predict(test_x.reshape(1, -1))[0]  # 测试数据，分类返回标记
            label_predict.loc[label_y0.index[d + para.month], 'label'] = predict_y
            if predict_y <= 1:
                ret = retdata.loc[first_day[d + para.month]:last_day[d + para.month], '中证500'] / 100  # 小盘
                label_predict.loc[label_y0.index[d + para.month], 'sml'] = 0
            else:
                ret = retdata.loc[first_day[d + para.month]:last_day[d + para.month], '沪深300'] / 100
                label_predict.loc[label_y0.index[d + para.month], 'sml'] = 1
            strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'return'] = ret

        strategy['nav'] = 0
        ret = strategy.loc[first_day[para.month]:last_day[para.month], 'return']
        strategy.loc[first_day[para.month]:last_day[para.month], 'nav'] = (1 + ret).cumprod() / (1 + para.trd_cost1)
        for d in range(1, len(label_predict)):
            ret = strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'return']
            if label_predict['sml'].iloc[d - 1] == label_predict['sml'].iloc[d]:
                strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'nav'] = strategy.loc[last_day[d + para.month - 1], 'nav'] * (
                    1 + ret).cumprod()
            else:
                strategy.loc[first_day[d + para.month]:last_day[d + para.month], 'nav'] = strategy.loc[last_day[d + para.month - 1], 'nav'] * (
                    1 + ret).cumprod() / (1 + para.trd_cost1) * (1 - para.trd_cost2)
        day_list = strategy.index
        estm.loc[str(i), 'ret1'] = ((1 + strategy['return']).cumprod()).iloc[-1] ** (
            1 / ((datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365)) - 1
        strategy = strategy.iloc[-700:] / strategy.iloc[-700]
        day_list = strategy.index
        estm.loc[str(i), 'ret'] = strategy['nav'].iloc[-1] ** (
            1 / ((datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365)) - 1
        estm.loc[str(i), 'chg1'] = str(i)
        estm.loc[str(i), 'mon'] = j
        label_predict = label_predict.loc[20150116:]
        estm.loc[str(i), 'p'] = (np.sum((label_predict['sml'] == 0) & (label_y0.loc[label_predict.index, 'label'] <= 1)) + np.sum(
            (label_predict['sml'] == 1) & (label_y0.loc[label_predict.index, 'label'] > 1))) / len(label_predict)
        # estm.loc[str(m1) + str(j1), 'ret1'] = ((1 + strategy['return']).cumprod()).iloc[-1] ** (
        #     1 / ((datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365)) - 1
        # strategy = strategy.iloc[-700:] / strategy.iloc[-700]
        # day_list = strategy.index
        # estm.loc[str(m1) + str(j1), 'ret'] = strategy['nav'].iloc[-1] ** (
        #     1 / ((datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365)) - 1
        # estm.loc[str(m1) + str(j1), 'chg1'] = j1
        # estm.loc[str(m1) + str(j1), 'mon'] = m1
        # label_predict = label_predict.loc[20150116:]
        # estm.loc[str(m1) + str(j1), 'p'] = (np.sum((label_predict['sml'] == 0) & (label_y0.loc[label_predict.index, 'label'] <= 1)) + np.sum(
        #     (label_predict['sml'] == 1) & (label_y0.loc[label_predict.index, 'label'] > 1))) / len(label_predict)
estm[estm['ret']==max(estm['ret'])]




plot_data = (1 + retdata.loc[strategy.index] / 100).cumprod()
plot_data['strategy(cost)'] = strategy['nav']
plot_data['strategy'] = (1 + strategy['return']).cumprod()
plot_data = plot_data.iloc[-700:] / plot_data.iloc[-700]
day_list = plot_data.index
plot_data.index = range(len(plot_data))

plt.figure(1)
day_delta = (datetime.strptime(str(day_list[-1]), '%Y%m%d') - datetime.strptime(str(day_list[0]), '%Y%m%d')).days / 365
plt.plot(range(len(plot_data)), plot_data['沪深300'], "midnightblue",
         label='沪深300：年化收益率 ' + str(round(100 * (plot_data['沪深300'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
plt.plot(range(len(plot_data)), plot_data['中证500'], "steelblue",
         label='中证500：年化收益率 ' + str(round(100 * (plot_data['中证500'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
plt.plot(range(len(plot_data)), plot_data['strategy'], "mistyrose",
         label='策略组合：年化收益率 ' + str(round(100 * (plot_data['strategy'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
plt.plot(range(len(plot_data)), plot_data['strategy(cost)'], "red",
         label='策略组合(加交易成本)：年化收益率 ' + str(round(100 * (plot_data['strategy(cost)'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
plt.xticks(range(0, len(plot_data), 180), day_list[range(0, len(plot_data), 180)])
plt.legend(loc='upper left')
plt.title('logistic回归')

print(np.sum(label_predict['label'] == label_y0.loc[label_predict.index, 'label']) / len(label_predict))
print((np.sum((label_predict['sml'] == 0) & (label_y0.loc[label_predict.index, 'label'] <= 1)) + np.sum(
    (label_predict['sml'] == 1) & (label_y0.loc[label_predict.index, 'label'] > 1))) / len(label_predict))
