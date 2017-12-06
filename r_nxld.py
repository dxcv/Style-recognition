import pandas as pd
from datetime import *
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.simplefilter(action="ignore", category=RuntimeWarning)
os.chdir('C:/Users/ryuan/Desktop/')
retdata = pd.read_excel('指数日涨幅.xlsx', index_col='时间')

retdata.index = [int(datetime.strptime(i, "%Y-%m-%d").strftime('%Y%m%d')) for i in retdata.index]

label_predict = pd.DataFrame(
    {'label': [0] * (len(retdata) - 20), 'label1': [0] * (len(retdata) - 20), 'sml': [0] * (len(retdata) - 20)},
    index=retdata.index[20:])
strategy = pd.DataFrame({'return': [0] * (len(retdata) - 20)}, index=retdata.index[20:])
for d in range(20,len(retdata)):
    hs300 = (1+retdata['沪深300'].iloc[(d-20):d]/100).cumprod().iloc[-1]
    zz500 = (1 + retdata['中证500'].iloc[(d-20):d] / 100).cumprod().iloc[-1]
    if hs300 < zz500:
        ret = retdata['中证500'].iloc[d] / 100  # 小盘
        label_predict['sml'].iloc[d-20] = 0
    else:
        ret = retdata['沪深300'].iloc[d] / 100
        label_predict['sml'].iloc[d-20] = 1
    if max(hs300,zz500)<1:
        ret = 0.03/200
        label_predict['sml'].iloc[d-20] = 2
    strategy.iloc[d-20] = ret

strategy['nav'] = 0
ret = strategy['return'].iloc[0]
strategy['nav'].iloc[0] = float((1 + ret) / (1 + 0.001))
for d in range(1, len(strategy)):
    ret = strategy['return'].iloc[d]
    if label_predict['sml'].iloc[d - 1] == label_predict['sml'].iloc[d]:
        strategy['nav'].iloc[d] = float(strategy['nav'].iloc[d-1] * (1 + ret))
    else:
        strategy['nav'].iloc[d] = float(strategy['nav'].iloc[d-1] * (1 + ret) / (1 + 0.001) * (1 - 0.001))

plot_data = (1 + retdata.loc[strategy.index] / 100).cumprod()
plot_data['strategy(cost)'] = strategy['nav']
plot_data['strategy'] = (1 + strategy['return']).cumprod()
plot_data = plot_data.iloc[-700:]/plot_data.iloc[-700]
day_list = plot_data.index
plot_data.index = range(len(plot_data))
# plot_data.plot()
# plt.figure(2)
p2 = plt.subplot(2,1,2)
day_delta = (datetime.strptime(str(day_list[-1]),'%Y%m%d') - datetime.strptime(str(day_list[0]),'%Y%m%d')).days/365
p2.plot(range(len(plot_data)), plot_data['沪深300'], "midnightblue",
        label='沪深300：年化收益率 ' + str(round(100 * (plot_data['沪深300'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
p2.plot(range(len(plot_data)), plot_data['中证500'], "steelblue",
        label='中证500：年化收益率 ' + str(round(100 * (plot_data['中证500'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
p2.plot(range(len(plot_data)), plot_data['strategy'], "mistyrose", label='策略组合：年化收益率 '+str(round(100 * (plot_data['strategy'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
p2.plot(range(len(plot_data)), plot_data['strategy(cost)'], "red", label='策略组合(加交易成本)：年化收益率 '+str(round(100 * (plot_data['strategy(cost)'].iloc[-1] ** (1 / day_delta) - 1), 2)) + '%')
plt.xticks(range(0, len(plot_data), 60), day_list[range(0, len(plot_data), 60)])
plt.legend(loc='upper left')
plt.title('牛熊轮动')


