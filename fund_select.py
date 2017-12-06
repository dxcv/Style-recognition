import numpy as np
import pandas as pd
from WindPy import *
from datetime import datetime as dt
import warnings
import os
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

w.start()  # 启动 Wind API
warnings.simplefilter(action="ignore", category=RuntimeWarning)
os.chdir('C:/Users/ryuan/Desktop/')
funddata = pd.read_excel('fund.xlsx')  # 这部分是筛选后的基金，均为主动型、开放型、非分级基金
trade_days = pd.read_excel('tradeday.xlsx')
trade_days = trade_days['TRADE_DAYS']
if not os.path.exists('Data1'):
    os.mkdir('Data1')
if not os.path.exists('Data2'):
    os.mkdir('Data2')

# 每季度的第一个交易日
tmp = np.unique(np.floor(trade_days[0:2144] / 100))
first_day = list()
for i in range(len(tmp)):
    first_day.append(min(trade_days[np.floor(trade_days / 100) == tmp[i]]))
last_day = list()
for i in range(len(tmp)):
    last_day.append(max(trade_days[np.floor(trade_days / 100) == tmp[i]]))
first_day = first_day[0::3]
last_day = last_day[2::3]
# 因子
factor = ['abs_ret1', 'abs_ret3', 'beta', 'calmar1', 'calmar3', 'down1', 'down3', 'loss1', 'loss3', 'mdd1',
          'mdd3', 'sharp1', 'sharp3', 'stock', 'timing', 'vol1', 'vol3', 'concentration', 'adjust', 'allocation', 'selecting', 'interaction']
factor_zh = ['abs_ret1', 'abs_ret3', 'beta', 'calmar1', 'calmar3', 'down1', 'down3', 'loss1', 'loss3', 'mdd1',
             'mdd3', 'sharp1', 'sharp3', 'stock', 'timing', 'vol1', 'vol3']
# 比较基准
code_bench = {'大盘成长': '399373.SZ', '大盘平衡': '399314.SZ', '大盘价值': '399372.SZ', '中盘成长': '399374.SZ', '中盘平衡': '399315.SZ',
              '中盘价值': '399375.SZ', '小盘成长': '399376.SZ', '小盘平衡': '399316.SZ', '小盘价值': '399377.SZ', '偏股基准': '000300.SH',
              '普通型': '057.CS', '激进型': '057.CS'}


def selection(day, fund):
    """
    筛选在指定日期满足FOF投资条件的偏股型基金
    :param day: 指定日期
    :param fund: 待筛选基金的代码及相关数据
    :return: number-基金数量，code-基金代码, operating_period-基金运营时间,asset-股票资产占比, type1-一级分类（股、债、混）, type2-二级分类（风格-价值、普通-激进、混）
    """
    daydate = dt.strptime(str(day), '%Y%m%d')
    operating_period = np.array([(daydate - dt.strptime(str(i), '%Y%m%d')).days for i in fund['setdate']])
    number = sum(operating_period > 365)
    code = fund['fundcode']
    code = np.array(code[operating_period > 365])
    operating_period = operating_period[operating_period > 365]
    # 最近的两个定期报告日
    if int(str(day)[4:]) < 400:
        startdate = str(int(str(day)[0:4]) - 1) + '0930'
        enddate = str(int(str(day)[0:4]) - 1) + '1231'
    elif int(str(day)[4:]) < 700:
        startdate = str(int(str(day)[0:4]) - 1) + '1231'
        enddate = str(day)[0:4] + '0331'
    elif int(str(day)[4:]) < 1000:
        startdate = str(day)[0:4] + '0331'
        enddate = str(day)[0:4] + '0630'
    else:
        startdate = str(day)[0:4] + '0630'
        enddate = str(day)[0:4] + '0930'
    for i in range(number):
        x = w.wss(code[i], "fund_stm_issuingdate_qty,prt_netasset", "rptDate=" + enddate + ";unit=1")
        if x.Data[1][0] > 0:
            break
    if x.Data[0][0] < daydate:
        x1 = w.wss(",".join(code), "prt_netasset,prt_stocktonav", "rptDate=" + enddate + ";unit=1")
        df1 = pd.DataFrame(x1.Data).T
        df1 = pd.DataFrame(np.column_stack((pd.DataFrame(x1.Codes), df1)))
        df = df1
        report_date = enddate
    else:
        x2 = w.wss(",".join(code), "prt_netasset,prt_stocktonav", "rptDate=" + startdate + ";unit=1")
        df2 = pd.DataFrame(x2.Data).T
        df2 = pd.DataFrame(np.column_stack((pd.DataFrame(x2.Codes), df2)))
        df = df2
        report_date = startdate
    number = sum(df.iloc[:, 1] > 1.e+08)
    code = pd.Series(np.array(code[df.iloc[:, 1] > 1.e+08]))
    operating_period = pd.Series(np.array(operating_period[df.iloc[:, 1] > 1.e+08]))
    asset = df[df.iloc[:, 1] > 1.e+08]
    asset = pd.Series(np.array(asset.iloc[:, 1]))
    rate = df[df.iloc[:, 1] > 1.e+08]
    rate = pd.Series(np.array(rate.iloc[:, 2]))
    type1 = pd.Series(number)
    type2 = pd.Series(number)
    idx = []
    for i in range(number):
        if rate[i] is None:
            type1[i] = None
            type2[i] = None
            continue
        if rate[i] > 60:
            type1[i] = '偏股型'
            idx.append(i)
            type2[i] = ''
        elif rate[i] < 40:
            type1[i] = '偏债型'
            if rate[i] > 20:
                type2[i] = '激进型'
            else:
                type2[i] = '普通型'
        else:
            type1[i] = '混合型'
            type2[i] = '混合型'
    x = w.wsd(','.join(code[idx]), "NAV_acc", "ED-6M", str(day), "Period=W")
    df = pd.DataFrame(x.Data)
    rp = np.log(np.array(df.iloc[:, 1:]) / np.array(df.iloc[:, 0:-1]))
    x = w.wsd("801813.SI,801811.SI,000058.SH,000057.SH", "close", "ED-6M", str(day), "Period=W")
    df1 = pd.DataFrame(x.Data)
    df1 = pd.DataFrame(np.log((np.array(df1.iloc[:, 1:]) / np.array(df1.iloc[:, 0:-1]))))
    df1.index = x.Codes
    Fs = df1.loc['801813.SI']  # 申万小盘：小盘
    Fb = df1.loc['801811.SI']  # 申万大盘：大盘
    Fv = df1.loc['000058.SH']  # 全指价值：价值
    Fg = df1.loc['000057.SH']  # 全指成长：成长
    tmp_type2 = [''] * len(df)
    for i in range(len(df)):
        x1 = np.column_stack((np.array(Fb), np.array(Fs)))
        y1 = np.array(rp[i, :])
        est = sm.OLS(y1, x1)
        est = est.fit()
        b_b, b_s = est.params  # 系数
        t1, t2 = est.tvalues
        # delta_e = np.std(est.resid)
        # est1 = sm.OLS(np.array(Fb), np.array(Fs))
        # est1 = est1.fit()
        # bk = est1.params
        # delta_k = np.std(est1.resid)
        # t1 = bk * delta_k * np.sqrt(len(Fb) - 2) / delta_e  # 大盘显著性
        # est2 = sm.OLS(np.array(Fs), np.array(Fb))
        # est2 = est2.fit()
        # bk = est2.params
        # delta_k = np.std(est2.resid)
        # t2 = bk * delta_k * np.sqrt(len(Fb) - 2) / delta_e
        if np.isnan(b_b):
            tmp_type2[i] = None
            continue
        if (t1 > 1.96) & (t2 > 1.96):
            if (b_b > 0.5) & (b_s < 0.5):
                str1 = '大盘'
            elif (b_b < 0.5) & (b_s > 0.5):
                str1 = '小盘'
            else:
                str1 = '中盘'
        elif (t1 < 1.96) & (t2 < 1.96):
            str1 = '中盘'
        elif t1 > 1.96:
            if b_b > 0.1:
                str1 = '大盘'
            else:
                str1 = '中盘'
        elif t2 > 1.96:
            if b_s > 0.1:
                str1 = '小盘'
            else:
                str1 = '中盘'
        x2 = np.column_stack((np.array(Fv), np.array(Fg)))
        y2 = np.array(rp[i, :])
        est = sm.OLS(y2, x2)
        est = est.fit()
        b_v, b_g = est.params
        t1, t2 = est.tvalues
        if (t1 > 1.96) & (t2 > 1.96):
            if (b_v > 0.5) & (b_g < 0.5):
                str2 = '价值'
            elif (b_v < 0.5) & (b_g > 0.5):
                str2 = '成长'
            else:
                str2 = '平衡'
        elif (t1 < 1.96) & (t2 < 1.96):
            str2 = '平衡'
        elif t1 > 1.96:
            if b_v > 0.1:
                str2 = '价值'
            else:
                str2 = '平衡'
        elif t2 > 1.96:
            if b_g > 0.1:
                str2 = '成长'
            else:
                str2 = '平衡'
        tmp_type2[i] = str1 + str2
    type2[idx] = tmp_type2
    for del_id in np.array(range(len(type2)))[[i is None for i in type2]]:
        del code[del_id]
        del operating_period[del_id]
        del asset[del_id]
        del type1[del_id]
        del type2[del_id]
    number = len(code)
    return number, code, operating_period, asset, type1, type2, report_date


def net(day, code):
    """
    指定日期前三个月的指定基金的净值数据
    :param day: 指定日期
    :param code: 指定基金代码
    :return: data-基金累计单位净值数据
    """
    enddate = str(day)[0:4] + '-' + str(day)[4:6] + '-' + str(day)[6:]
    x = w.wsd(",".join(code), "NAV_acc", "ED-3M", enddate, "")
    data = pd.DataFrame(x.Data).T
    data.columns = code
    data.index = x.Times
    return data


def available(day, code, data):
    """
    检查数据是否符合要求
    :param day: 指定日期
    :param code: 指定基金代码
    :param data: 基金净值数据
    :return: 符合要求的数据
    """
    data1 = data[[list(data.columns).index(i) for i in code]]
    if day not in [int(i.strftime('%Y%m%d')) for i in list(data.index)]:
        print('ATTENTION: %d is not in the data list' % day)
        return
    if data is None:
        print('ATTENTION: data is none')
        return
    if not isinstance(data, pd.DataFrame):
        print('ATTENTION: type of data is ' + str(type(data)) + ', please input data as DataFrame')
        return
    if (list(data.index)[-1] - list(data.index)[0]).days < 80:
        print('ATTENTION: time interval is %d days, too short' % (list(data.index)[-1] - list(data.index)[0]).days)
        return
    return data1


def date_index(day, data):
    """
    数据对应的index
    :param day: 指定日期
    :param data: 净值数据
    :return: day1m_index-一个月前的位置, day3m_index-三个月前的位置, day_index-当日位置
    """
    day1m = dt.strptime(str(day), '%Y%m%d') - relativedelta(months=+1)
    temp = np.array([int(i.strftime('%Y%m%d')) for i in list(data.index)])
    temp = temp[temp >= int(day1m.strftime('%Y%m%d'))][0]
    day1m_index = [int(i.strftime('%Y%m%d')) for i in list(data.index)].index(temp)
    day3m = dt.strptime(str(day), '%Y%m%d') - relativedelta(months=+3)
    temp = np.array([int(i.strftime('%Y%m%d')) for i in list(data.index)])
    temp = temp[temp >= int(day3m.strftime('%Y%m%d'))][0]
    day3m_index = [int(i.strftime('%Y%m%d')) for i in list(data.index)].index(temp)
    day_index = [int(i.strftime('%Y%m%d')) for i in list(data.index)].index(day) + 1
    return day1m_index, day3m_index, day_index


def abs_ret(day, code, data):
    """
    收益指标：绝对收益
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :return: abs_ret1-过去一个月的净值收益, abs_ret3-过去三个月的净值收益
    """
    data = available(day, code, data)
    day1m_index, day3m_index, day_index = date_index(day, data)
    abs_ret1 = data.iloc[day_index - 1, :] / data.iloc[day1m_index, :] - 1.
    abs_ret2 = data.iloc[day_index - 1, :] / data.iloc[day3m_index, :] - 1.
    return abs_ret1, abs_ret2


def risk(day, code, data, bench):
    """
    风险指标：波动率、最大回撤、亏损频率、下行风险
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :return: vol1-过去一个月的波动率, vol3-过去三个月的波动率, mdd1-过去一个月的最大回撤, mdd3-过去三个月的最大回撤, loss1-过去一个月的亏损频率, loss3-过去三个月的亏损频率, down1-过去一个月的下行风险, down3-过去三个月的下行风险
    """
    data = available(day, code, data)
    r = pd.DataFrame(np.array(data.iloc[1:, :]) / np.array(data.iloc[0:-1, :]) - 1.)
    r.columns = data.columns
    r.index = data.index[1:]
    day1m_index, day3m_index, day_index = date_index(day, r)
    vol1 = np.std(r.iloc[day1m_index:day_index, :])
    vol3 = np.std(r.iloc[day3m_index:day_index, :])
    loss1 = np.sum(r.iloc[day1m_index:day_index, :] < 0, axis=0) / r.iloc[day1m_index:day_index, :].shape[0]
    loss3 = np.sum(r.iloc[day3m_index:day_index, :] < 0, axis=0) / r.iloc[day3m_index:day_index, :].shape[0]
    rm = np.array(bench.iloc[1:, :]) / np.array(bench.iloc[0:-1, :]) - 1.
    rp = r.iloc[day1m_index:day_index, :] - np.tile(rm[day1m_index:day_index], r.shape[1])
    down1 = np.zeros(r.shape[1])
    for i in range(r.shape[1]):
        tmpr = rp.iloc[:, i]
        tmpr = tmpr[tmpr < 0]
        down1[i] = np.power(sum(np.power(tmpr, 2)) / (len(tmpr) - 1), 1 / 2)
    down1 = pd.Series(down1)
    down1.index = vol1.index
    rp = r.iloc[day3m_index:day_index, :] - np.tile(rm[day3m_index:day_index], r.shape[1])
    down3 = np.zeros(r.shape[1])
    for i in range(r.shape[1]):
        tmpr = rp.iloc[:, i]
        tmpr = tmpr[tmpr < 0]
        down3[i] = np.power(sum(np.power(tmpr, 2)) / (len(tmpr) - 1), 1 / 2)
    down3 = pd.Series(down3)
    down3.index = vol1.index
    day1m_index, day3m_index, day_index = date_index(day, data)
    values = data.iloc[day1m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    mdd1 = abs(np.min(dd, axis=0))
    values = data.iloc[day3m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    mdd3 = abs(np.min(dd, axis=0))
    return vol1, vol3, mdd1, mdd3, loss1, loss3, down1, down3


def risk_ret(day, code, data):
    """
    风险调整后收益指标：夏普比率、卡玛比率
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :return: sharp1-过去一个月的夏普比率, sharp1-过去三个月的夏普比率, calmar1-过去一个月的卡玛比率, calmar3-过去三个月的卡玛比率
    """
    data = available(day, code, data)
    r = pd.DataFrame(np.array(data.iloc[1:, :]) / np.array(data.iloc[0:-1, :]) - 1.)
    r.columns = data.columns
    r.index = data.index[1:]
    day1m_index, day3m_index, day_index = date_index(day, r)
    sharp1 = np.mean(r.iloc[day1m_index:day_index, :]) / np.std(r.iloc[day1m_index:day_index, :])
    sharp3 = np.mean(r.iloc[day3m_index:day_index, :]) / np.std(r.iloc[day3m_index:day_index, :])
    day1m_index, day3m_index, day_index = date_index(day, data)
    values = data.iloc[day1m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    calmar1 = (values.iloc[-1] / values.iloc[0] - 1) / values.shape[0] * 244 / abs(np.min(dd, axis=0))
    values = data.iloc[day3m_index:day_index, :]
    dd = pd.DataFrame([values.iloc[i, :] / values.iloc[0:i + 1, :].max() - 1 for i in range(len(values))])
    calmar3 = (values.iloc[-1] / values.iloc[0] - 1) / values.shape[0] * 244 / abs(np.min(dd, axis=0))
    return sharp1, sharp3, calmar1, calmar3


def cl_model(day, code, data, bench):
    """
    C-L模型指标：选股能力、择时能力、系统风险系数
    :param day: 截面日期
    :param code: 指定基金代码
    :param data: 基金累计单位净值数据
    :param bench: 比较基准
    :return: stock-选股能力, timing-择时能力, beta-系统风险系数
    """
    data = available(day, code, data)
    r = pd.DataFrame(np.array(data.iloc[1:, :]) / np.array(data.iloc[0:-1, :]) - 1.)
    r.columns = data.columns
    r.index = data.index[1:]
    day1m_index, day3m_index, day_index = date_index(day, r)
    rf = 4 / 100 / 365
    rm = (np.array(bench.iloc[1:, :]) / np.array(bench.iloc[0:-1, :]) - 1.)[day3m_index:day_index]
    rp = r.iloc[day3m_index:day_index, :]
    stock = pd.Series([None] * (rp.shape[1]))
    timing = pd.Series([None] * (rp.shape[1]))
    beta = pd.Series([None] * (rp.shape[1]))
    for i in range(rp.shape[1]):
        x = np.column_stack((np.array([float(max(aa, 0)) for aa in (rm - rf)]), np.array([float(min(aa, 0)) for aa in (rm - rf)])))
        x = sm.add_constant(x)
        y = np.array(rp.iloc[:, i]) - rf
        est = sm.OLS(y, x)
        est = est.fit()
        stock[i], beta[i], timing[i] = est.params
    stock.index = data.columns
    timing.index = data.columns
    beta.index = data.columns
    return stock, timing, beta


def position(code_now, dflist, sector_struct, rptdate):
    concentration = [None] * len(code_now)
    adjust = [None] * len(code_now)
    allocation = [None] * len(code_now)
    selecting = [None] * len(code_now)
    interaction = [None] * len(code_now)
    code1 = list(set(code_now).intersection(set(dflist.keys())))
    code_idx = [list(code_now).index(ai) for ai in code1]
    if code1 == []:
        return concentration, adjust, allocation, selecting, interaction
    else:
        for j in code_idx:
            i = code_now.iloc[j]
            tmp_df1 = dflist[i]
            if rptdate[4:6] == '03':
                lstdate = str(int(rptdate) - 9100)
            elif rptdate[4:6] == '06':
                lstdate = str(int(rptdate) - 299)
            elif rptdate[4:6] == '09':
                lstdate = str(int(rptdate) - 300)
            elif rptdate[4:6] == '12':
                lstdate = str(int(rptdate) - 301)
            if int(rptdate) not in list(tmp_df1['截止日']):
                concentration = None
                adjust = None
                continue
            tmp_df = tmp_df1.loc[tmp_df1['截止日'] == int(rptdate)]
            concentration[j] = sum(tmp_df.loc[(tmp_df1['sort_id'] <= 10), '占净值比例(%)']) / 100
            remain = list(set(tmp_df['code']).intersection(set(tmp_df1.loc[(tmp_df1['sort_id'] <= 10) & (tmp_df1['截止日'] == int(lstdate)), 'code'])))
            adjust[j] = 1 - len(remain) / len(tmp_df)
            if (rptdate[4:6] == '03') | (rptdate[4:6] == '09'):
                bdate = lstdate
            else:
                bdate = rptdate
            bdf = tmp_df1.loc[tmp_df1['截止日'] == int(bdate)]
            bdf = bdf.dropna()
            if bdf.shape[0] == 0:
                continue
            x = w.wss(','.join(bdf['code']), "pct_chg", "tradeDate=" + bdate + ";cycle=M")
            if x.ErrorCode != 0:
                continue
            bdf['pct_chg'] = np.array(x.Data[0]) / 100
            bdf['weights'] = bdf['占净值比例(%)'] / sum(bdf['占净值比例(%)'])
            bdf = bdf.dropna()
            if bdf.shape[0] == 0:
                continue
            bdf = bdf.drop(bdf[bdf['weights'] == 0].index)
            if bdf.shape[0] == 0:
                continue
            grouped = bdf.groupby('sector')
            get_wavg = lambda g: np.average(g['pct_chg'], weights=g['weights'])
            bdftmp = pd.DataFrame()
            bdftmp['rp'] = grouped.apply(get_wavg)
            bdftmp['wp'] = grouped.apply(lambda g: np.mean(g['weights']))
            bdftmp['sector'] = grouped.apply(get_wavg).index
            bench_df = pd.merge(sector_struct, bdftmp, how='inner', on='sector')
            bench_df = bench_df.rename(columns={'中信行业代码': 'sector_code', '权重': 'wb'})
            x = w.wss(','.join(bench_df['sector_code']), "pct_chg", "tradeDate=" + bdate + ";cycle=M")
            if x.ErrorCode != 0:
                continue
            bench_df['rb'] = np.array(x.Data[0]) / 100
            allocation[j] = sum((bench_df['wp'] - bench_df['wb']) * bench_df['rb'])
            selecting[j] = sum((bench_df['rp'] - bench_df['rb']) * bench_df['wb'])
            interaction[j] = sum((bench_df['rp'] - bench_df['rb']) * (bench_df['wp'] - bench_df['wb']))
    return concentration, adjust, allocation, selecting, interaction


all_bsc = pd.DataFrame(
    {'大盘成长': [0] * (len(first_day)), '大盘平衡': [0] * (len(first_day)), '大盘价值': [0] * (len(first_day)), '中盘成长': [0] * (len(first_day)),
     '中盘平衡': [0] * (len(first_day)), '中盘价值': [0] * (len(first_day)), '小盘成长': [0] * (len(first_day)), '小盘平衡': [0] * (len(first_day)),
     '小盘价值': [0] * (len(first_day)), '普通': [0] * (len(first_day)), '激进': [0] * (len(first_day)), '混合': [0] * (len(first_day))})
all_bsc.index = first_day
for d in reversed(np.array(first_day)):
    print(d)
    if not os.path.exists('Data1/' + str(d)):
        os.mkdir('Data1/' + str(d))
    number, code_all, operating_period, asset, type1, type2, report_date = selection(d, funddata)
    if number <= 0:
        print('ERROR: no data!')
        break
    df_basic = pd.DataFrame(
        {'number': number, 'code': code_all, 'operating_period': operating_period, 'asset': asset, 'type1': type1, 'type2': type2, 'report_date': report_date})
    df_basic.to_csv('Data1/' + str(d) + '/basic.csv', index=False)
    x = df_basic.groupby(by='type2')
    x1 = x.count()['code']
    for j in x1.index:
        c = [n for n in all_bsc.columns if n in j][0]
        all_bsc.loc[int(d), c] = x1[j]
all_bsc.to_csv('fundnumber.csv')

sector = pd.read_excel('C:/Users/ryuan/Desktop/中信一级行业170724.xlsx')
# 遍历指定目录，显示目录下的所有文件名
filepath = 'C:/Users/ryuan/Desktop/data/'
pathDir = os.listdir(filepath)
dflist = {}
for allDir in pathDir:
    print(allDir)
    temp = pd.read_csv(filepath + allDir, encoding='gbk')
    if len(temp) == 0:
        continue
    temp['sector'] = None
    temp['code'] = None
    for i in range(len(temp['代码'])):
        code_tmp = '0' * (6 - len(str(temp['代码'].iloc[i]))) + str(temp['代码'].iloc[i])
        if code_tmp[0] == '6':
            code_tmp = code_tmp + '.SH'
        else:
            code_tmp = code_tmp + '.SZ'
        temp.loc[i, 'code'] = code_tmp
        x = np.array(sector[sector['证券代码'].isin([temp['代码'].iloc[i]])])
        if len(x) == 0:
            continue
        temp.loc[i, 'sector'] = x[0, 2]
    temp['sort_id'] = temp['占净值比例(%)'].groupby(temp['截止日']).rank(ascending=False)
    name_tmp = allDir[2:8] + '.OF'
    dflist[name_tmp] = temp
sector_struct = pd.read_excel('沪深300指数行业占比.xlsx')
sector_struct = sector_struct.rename(columns={'中信行业名称': 'sector'})
sector_struct['权重'] = sector_struct['权重'] / 100

for d in reversed(np.array(first_day)):
    print(d)
    df_basic = pd.read_csv('Data1/' + str(d) + '/basic.csv', encoding="gb2312")
    number = df_basic.loc[0, 'number']
    code_all = df_basic['code']
    operating_period = df_basic['operating_period']
    asset = df_basic['asset']
    type1 = df_basic['type1']
    type2 = df_basic['type2']
    report_date = df_basic.loc[0, 'report_date']
    # # 同期沪深300净值
    # x = w.wsd("000300.SH", "close", "ED-3M", str(d), "")
    # hs300 = pd.DataFrame(x.Data[0])
    # hs300.index = x.Times
    # hs300.to_csv('Data1/' + str(d) + '/hs300net.csv')
    for i in np.unique(type2):
        tmp_dir = 'Data1/' + str(d) + '/' + list(type1[type2 == i])[0] + '/' + i
        if not os.path.exists('Data1/' + str(d) + '/' + list(type1[type2 == i])[0]):
            os.mkdir('Data1/' + str(d) + '/' + list(type1[type2 == i])[0])
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        code = code_all[type2 == i]
        # data = net(d, code)
        # data.to_csv(tmp_dir + '/netvalue.csv')
        # if hs300.shape[0] != data.shape[0]:
        #     print('ERROR: length not pair!')
        #     break
        # abs_ret1, abs_ret2 = abs_ret(d, code, data)
        # df_ret = pd.DataFrame({'abs_ret1': abs_ret1, 'abs_ret2': abs_ret2})
        # df_ret.to_csv(tmp_dir + '/absret.csv')
        df_ret = pd.read_csv(tmp_dir + '/absret.csv')
        abs_ret1, abs_ret2 = df_ret['abs_ret1'], df_ret['abs_ret2']
        # vol1, vol3, mdd1, mdd3, loss1, loss3, down1, down3 = risk(d, code, data, hs300)
        # df_risk = pd.DataFrame({'vol1': vol1, 'vol3': vol3, 'mdd1': mdd1, 'mdd3': mdd3, 'loss1': loss1, 'loss3': loss3, 'down1': down1, 'down3': down3})
        # df_risk.to_csv(tmp_dir + '/risk.csv')
        df_risk = pd.read_csv(tmp_dir + '/risk.csv')
        vol1, vol3, mdd1, mdd3, loss1, loss3, down1, down3 = df_risk['vol1'], df_risk['vol3'], df_risk['mdd1'], df_risk['mdd3'], df_risk['loss1'], df_risk[
            'loss3'], df_risk['down1'], df_risk['down3']
        # sharp1, sharp3, calmar1, calmar3 = risk_ret(d, code, data)
        # df_riskret = pd.DataFrame({'sharp1': sharp1, 'sharp3': sharp3, 'calmar1': calmar1, 'calmar3': calmar3})
        # df_riskret.to_csv(tmp_dir + '/riskret.csv')
        df_riskret = pd.read_csv(tmp_dir + '/riskret.csv')
        sharp1, sharp3, calmar1, calmar3 = df_riskret['sharp1'], df_riskret['sharp3'], df_riskret['calmar1'], df_riskret['calmar3']
        # stock, timing, beta = cl_model(d, code, data, hs300)
        # df_cl = pd.DataFrame({'stock': stock, 'timing': timing, 'beta': beta})
        # df_cl.to_csv(tmp_dir + '/clmodel.csv')
        df_cl = pd.read_csv(tmp_dir + '/clmodel.csv')
        stock, timing, beta = df_cl['stock'], df_cl['timing'], df_cl['beta']
        if i in ['大盘成长', '大盘平衡', '大盘价值', '中盘成长', '中盘平衡', '中盘价值', '小盘成长', '小盘平衡', '小盘价值']:
            concentration, adjust, allocation, selecting, interaction = position(code, dflist, sector_struct, str(report_date))
            df_pos = pd.DataFrame(
                {'concentration': concentration, 'adjust': adjust, 'allocation': allocation, 'selecting': selecting, 'interaction': interaction})
            df_pos.to_csv(tmp_dir + '/position.csv')
        else:
            concentration = [None] * len(code)
            adjust = [None] * len(code)
            allocation = [None] * len(code)
            selecting = [None] * len(code)
            interaction = [None] * len(code)
            df_pos = pd.DataFrame(
                {'concentration': concentration, 'adjust': adjust, 'allocation': allocation, 'selecting': selecting, 'interaction': interaction})
            df_pos.to_csv(tmp_dir + '/position.csv')
        all_x = pd.DataFrame(
            {'abs_ret1': abs_ret1, 'abs_ret3': abs_ret2, 'vol1': vol1, 'vol3': vol3, 'mdd1': mdd1,
             'mdd3': mdd3, 'loss1': loss1, 'loss3': loss3, 'down1': down1, 'down3': down3, 'sharp1': sharp1, 'sharp3': sharp3, 'calmar1': calmar1,
             'calmar3': calmar3, 'stock': stock, 'timing': timing, 'beta': beta, 'concentration': concentration, 'adjust': adjust, 'allocation': allocation,
             'selecting': selecting, 'interaction': interaction})
        if not os.path.exists('Data2/' + list(type1[type2 == i])[0]):
            os.mkdir('Data2/' + list(type1[type2 == i])[0])
        if not os.path.exists('Data2/' + list(type1[type2 == i])[0] + '/' + i):
            os.mkdir('Data2/' + list(type1[type2 == i])[0] + '/' + i)
        all_x.to_csv('Data2/' + list(type1[type2 == i])[0] + '/' + i + '/' + str(d) + 'X.csv')

for i in os.listdir('Data2'):
    print(i)
    for j in os.listdir('Data2/' + i):
        print(j)
        ic_tmp = np.zeros([len(factor), len(first_day) - 1])
        for d in range(len(first_day) - 1):
            fd = first_day[d]
            ld = last_day[d]
            print(fd)
            if os.path.exists('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv'):
                x = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv', index_col=0)
                code = x.index
                yfd = w.wsd(",".join(code), "NAV_acc", str(fd), str(fd), "")
                yld = w.wsd(",".join(code), "NAV_acc", str(ld), str(ld), "")
                logret = pd.DataFrame({'code': code, 'Y': np.log(np.array(yld.Data[0]) / np.array(yfd.Data[0]))})
                logret.to_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'Y.csv')

for i in os.listdir('Data2'):
    print(i)
    ic_tmp = np.zeros([len(factor), len(first_day) - 1])
    for d in range(len(first_day) - 1):
        fd = first_day[d]
        ld = last_day[d]
        print(fd)
        x = pd.DataFrame()
        y = pd.DataFrame()
        for j in os.listdir('Data2/' + i):
            print(j)
            if os.path.exists('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv'):
                x_tmp = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv', index_col=0)
                y_tmp = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'Y.csv', index_col=2)
                x = x.append(x_tmp)
                y = y.append(y_tmp)
        for k in range(len(factor)):
            tmp_fct = factor[k]
            a1 = pd.DataFrame(x[tmp_fct]).reset_index()
            a2 = pd.DataFrame(y['Y']).reset_index()
            idx1 = a1[np.isnan(a1[tmp_fct])].index
            idx2 = a2[np.isnan(a2['Y'])].index
            a1 = np.array(a1.drop(idx1 | idx2)[tmp_fct])
            a2 = np.array(a2.drop(idx1 | idx2)['Y'])
            ic_tmp[k, d] = np.corrcoef(a1, a2)[0][1]
    ic_df = pd.DataFrame(ic_tmp, index=factor, columns=first_day[0:-1])
    ic_df.to_csv(i + 'ictest.csv')

weight1 = pd.read_excel('IC加权.xlsx', sheetname='Sheet1')
weight2 = pd.read_excel('IC加权.xlsx', sheetname='Sheet2')
all_stg = pd.DataFrame(
    {'大盘成长': [0] * (len(first_day) - 1), '大盘平衡': [0] * (len(first_day) - 1), '大盘价值': [0] * (len(first_day) - 1), '中盘成长': [0] * (len(first_day) - 1),
     '中盘平衡': [0] * (len(first_day) - 1), '中盘价值': [0] * (len(first_day) - 1), '小盘成长': [0] * (len(first_day) - 1), '小盘平衡': [0] * (len(first_day) - 1),
     '小盘价值': [0] * (len(first_day) - 1), '普通型': [0] * (len(first_day) - 1), '激进型': [0] * (len(first_day) - 1), '混合型': [0] * (len(first_day) - 1)})
all_stg_s = pd.DataFrame(
    {'大盘成长': [''] * (len(first_day) - 1), '大盘平衡': [''] * (len(first_day) - 1), '大盘价值': [''] * (len(first_day) - 1), '中盘成长': [''] * (len(first_day) - 1),
     '中盘平衡': [''] * (len(first_day) - 1), '中盘价值': [''] * (len(first_day) - 1), '小盘成长': [''] * (len(first_day) - 1), '小盘平衡': [''] * (len(first_day) - 1),
     '小盘价值': [''] * (len(first_day) - 1), '普通型': [''] * (len(first_day) - 1), '激进型': [''] * (len(first_day) - 1), '混合型': [''] * (len(first_day) - 1)})
all_stg_n = pd.DataFrame(
    {'大盘成长': [0] * (len(first_day) - 1), '大盘平衡': [0] * (len(first_day) - 1), '大盘价值': [0] * (len(first_day) - 1), '中盘成长': [0] * (len(first_day) - 1),
     '中盘平衡': [0] * (len(first_day) - 1), '中盘价值': [0] * (len(first_day) - 1), '小盘成长': [0] * (len(first_day) - 1), '小盘平衡': [0] * (len(first_day) - 1),
     '小盘价值': [0] * (len(first_day) - 1), '普通型': [0] * (len(first_day) - 1), '激进型': [0] * (len(first_day) - 1), '混合型': [0] * (len(first_day) - 1)})
all_coef = pd.DataFrame(
    {'大盘成长': [0] * (len(first_day) - 1), '大盘平衡': [0] * (len(first_day) - 1), '大盘价值': [0] * (len(first_day) - 1), '中盘成长': [0] * (len(first_day) - 1),
     '中盘平衡': [0] * (len(first_day) - 1), '中盘价值': [0] * (len(first_day) - 1), '小盘成长': [0] * (len(first_day) - 1), '小盘平衡': [0] * (len(first_day) - 1),
     '小盘价值': [0] * (len(first_day) - 1), '普通型': [0] * (len(first_day) - 1), '激进型': [0] * (len(first_day) - 1), '混合型': [0] * (len(first_day) - 1)})
all_last5 = pd.DataFrame(
    {'大盘成长': [0] * (len(first_day) - 1), '大盘平衡': [0] * (len(first_day) - 1), '大盘价值': [0] * (len(first_day) - 1), '中盘成长': [0] * (len(first_day) - 1),
     '中盘平衡': [0] * (len(first_day) - 1), '中盘价值': [0] * (len(first_day) - 1), '小盘成长': [0] * (len(first_day) - 1), '小盘平衡': [0] * (len(first_day) - 1),
     '小盘价值': [0] * (len(first_day) - 1), '普通型': [0] * (len(first_day) - 1), '激进型': [0] * (len(first_day) - 1), '混合型': [0] * (len(first_day) - 1)})
all_score = []
all_y = []
for i in os.listdir('Data2'):
    if i == '偏股型':
        a = weight1['偏股型'] * weight2['偏股型']
        a.index = factor
        a = a.loc[factor_zh]
    elif i == '偏债型':
        a = weight1['偏债型'] * weight2['偏债型']
        a.index = factor
        a = a.loc[factor_zh]
    else:
        a = weight1['混合型'] * weight2['混合型']
        a.index = factor
        a = a.loc[factor_zh]
    for j in os.listdir('Data2/' + i):
        strategy = pd.DataFrame({'return': [0] * (len(first_day) - 1), 'selection': [''] * (len(first_day) - 1), 'number': [0] * (len(first_day) - 1)})
        score_coef = np.zeros(len(first_day) - 1)
        last5 = np.zeros(len(first_day) - 1)
        c = j
        for d in range(len(first_day) - 1):
            fd = first_day[d]
            if os.path.exists('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv'):
                x = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv')
                y = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'Y.csv', index_col=0)
                if i == '偏股型':
                    # x = x.loc[:, factor]
                    x = x.loc[:, factor_zh]
                else:
                    x = x.loc[:, factor_zh]
                idx1 = x[np.sum(np.isinf(x), axis=1) > 0].index
                idx2 = x[np.sum(np.isnan(x), axis=1) > 0].index
                x = x.drop(idx1 | idx2)
                y = y.drop(idx1 | idx2).reset_index(drop=True)
                scaler = preprocessing.StandardScaler().fit(x)
                x = scaler.transform(x)
                score = np.dot(x, a)
                score = (score - min(score)) / (max(score) - min(score)) * 10
                y_score_curr_month = pd.DataFrame(score)
                y_score_curr_month = y_score_curr_month.sort_values(by=0, ascending=False)
                n_stock_select = 5  # int(np.ceil(len(x) / 10))
                if len(y_score_curr_month) < n_stock_select:
                    n_stock_select = len(y_score_curr_month)
                index_select = y_score_curr_month.iloc[0:n_stock_select].index
                strategy.loc[d, 'return'] = np.mean(np.exp(y.loc[index_select, 'Y'])) - 1
                strategy.loc[d, 'selection'] = '、'.join(y.loc[index_select, 'code'])
                strategy.loc[d, 'number'] = len(y.loc[index_select, 'code'])
                score_coef[d] = np.corrcoef(np.array(score), np.array(y['Y']))[0][1]
                all_score.extend(score.tolist())
                all_y.extend((y['Y']).tolist())
                last5[d] = np.mean(np.exp(y.loc[y_score_curr_month.iloc[-n_stock_select:].index, 'Y'])) - 1
        all_stg[c] = (strategy['return'] + 1).cumprod()
        all_stg_s[c] = strategy['selection']
        all_stg_n[c] = strategy['number']
        all_coef[c] = score_coef
        all_last5[c] = (last5 + 1).cumprod()
# plt.plot(np.array(all_score),np.array(all_y),'ro')
all_stg.index = first_day[1:]
all_stg.loc[first_day[0]] = 1
all_stg = all_stg.sort_index(ascending=1)
all_stg_s.index = first_day[1:]
all_stg_n.index = first_day[1:]
all_last5.index = first_day[1:]
all_last5.loc[first_day[0]] = 1
all_last5 = all_last5.sort_index(ascending=1)
tmp = []
for i in all_stg.index:
    tmp.append(str(i)[0:4] + '-' + str(i)[4:6] + '-' + str(i)[6:8])
all_stg.index = tmp
all_stg.to_csv('strategy.csv')
all_last5.index = tmp
all_last5.to_csv('strategy_last5.csv')

all_stg = pd.read_csv('strategy.csv',index_col=0,encoding='gbk')
all_stg.index=first_day
all_last5 = pd.read_csv('strategy_last5.csv',index_col=0,encoding='gbk')
all_last5.index=first_day
plt.style.use('ggplot')
k = 0
for i in list(all_last5.keys()):
    stg_tmp = all_stg[i]
    r = pd.DataFrame(np.array(stg_tmp[1:]) / np.array(stg_tmp[0:-1]) - 1)
    r.index = stg_tmp.index[1:]
    stg_ac = pd.DataFrame(stg_tmp)
    stg_ac.iloc[1] = float(1 / (1 + 0.006) * (1 + r.iloc[0]) * (1 - 0.005))  # 申购费千分之六, 赎回费千分之五
    code_tmp = all_stg_s.loc[first_day[1], i].split('、')
    for d in range(2, len(all_stg)):
        dd = stg_tmp.index[d]
        remainlist = list(set(all_stg_s.loc[dd, i].split('、')).intersection(set(code_tmp)))
        hsl = 1 - len(remainlist) / len(code_tmp)
        stg_ac.iloc[d] = float(float(stg_ac.iloc[d - 1]) / (1 + hsl * 0.006) * (1 + r.iloc[d - 1]) * (1 - hsl * 0.005))
        code_tmp = all_stg_s.loc[dd, i].split('、')
    nav_p = stg_ac
    stg_tmp1 = all_last5[i]
    r1 = pd.DataFrame(np.array(stg_tmp1[1:]) / np.array(stg_tmp1[0:-1]) - 1)
    r1.index = stg_tmp1.index[1:]
    stg_ac1 = pd.DataFrame(stg_tmp1)
    stg_ac1.iloc[1] = float(1 / (1 + 0.006) * (1 + r1.iloc[0]) * (1 - 0.005))  # 申购费千分之六, 赎回费千分之五
    for d in range(2, len(all_last5)):
        stg_ac1.iloc[d] = float(float(stg_ac1.iloc[d - 1]) / (1 + 0.006) * (1 + r1.iloc[d - 1]) * (1 - 0.005))
    nav_b1 = stg_ac1
    nav_b3 = w.wsd('000300.SH', "close", '2009-01-01', "2017-10-10", "Period=Q").Data[0]
    nav_b3 = np.array(nav_b3) / nav_b3[0]
    if i == '混合型':
        nav_b21 = nav_b3
        nav_b22 = w.wsd('057.CS', "close", '2009-01-01', "2017-10-10", "Period=Q").Data[0]
        nav_b2 = 0.5 * (nav_b21) + 0.5 * (np.array(nav_b22) / nav_b22[0])
    else:
        nav_b2 = w.wsd(code_bench[i], "close", '2009-01-01', "2017-10-10", "Period=Q").Data[0]
        nav_b2 = np.array(nav_b2) / nav_b2[0]
    plt.figure(k, figsize=(10, 8))
    plt.plot(range(len(nav_p)), nav_p, "red", label="基金组合")
    plt.plot(range(len(nav_b1)), nav_b1, "g", label="最后5名组合")
    plt.plot(range(len(nav_b2)), nav_b2, "b--", label="比较基准")
    plt.plot(range(len(nav_b3)), nav_b3, "c--", label="沪深300")
    plt.xticks(range(0, len(nav_b2), 10), nav_p.index[range(0, len(nav_b2), 10)])
    plt.legend(loc='upper left')
    plt.savefig('fig/' + i + ".jpg")
    k = k + 1

# ac_stg_all = pd.DataFrame(
#     {'大盘成长': [0] * (len(first_day)), '大盘平衡': [0] * (len(first_day)), '大盘价值': [0] * (len(first_day)), '中盘成长': [0] * (len(first_day)),
#      '中盘平衡': [0] * (len(first_day)), '中盘价值': [0] * (len(first_day)), '小盘成长': [0] * (len(first_day)), '小盘平衡': [0] * (len(first_day)),
#      '小盘价值': [0] * (len(first_day)), '普通型': [0] * (len(first_day)), '激进型': [0] * (len(first_day)), '混合型': [0] * (len(first_day))})
# ac_stg_all.index = all_stg.index
# for i in range(all_stg.shape[1]):
#     tmp = all_stg.columns[i]
#     stg_tmp = all_stg[tmp]
#     if all_stg_n.iloc[-1, i] == 0:
#         continue
#     if sum(all_stg_n[tmp] == 0) > 0:
#         for j in reversed(all_stg_n.index):
#             if (all_stg_n.loc[j, tmp] == 0) & (list(all_stg_s.index).index(j) < len(all_stg_s) - 1):
#                 sd = all_stg_n.index[list(all_stg_s.index).index(j) + 1]
#                 break
#     else:
#         sd = all_stg_n.index[0]
#     stg_tmp = stg_tmp.loc[all_stg.index[list(all_stg_s.index).index(sd)]:] / stg_tmp.loc[all_stg.index[list(all_stg_s.index).index(sd)]]
#     code_tmp = all_stg_s.loc[sd, tmp].split('、')
#     r = pd.DataFrame(np.array(stg_tmp[1:]) / np.array(stg_tmp[0:-1]) - 1)
#     r.index = stg_tmp.index[1:]
#     stg_ac = pd.DataFrame(stg_tmp)
#     stg_ac.loc[sd] = float(1 / (1 + 0.006) * (1 + r.loc[sd]) * (1 - 0.005))  # 申购费千分之六, 赎回费千分之五
#     for d in range(list(all_stg_s.index).index(sd) + 1, len(all_stg_s)):
#         remainlist = list(set(all_stg_s.loc[all_stg_s.index[d], tmp].split('、')).intersection(set(code_tmp)))
#         hsl = 1 - len(remainlist) / len(code_tmp)
#         stg_ac.loc[all_stg_s.index[d]] = float(
#             float(stg_ac.loc[all_stg_s.index[d - 1]]) / (1 + hsl * 0.006) * (1 + r.loc[all_stg_s.index[d]]) * (1 - hsl * 0.005))
#         code_tmp = all_stg_s.loc[all_stg_s.index[d], tmp].split('、')
#     ac_stg_all[tmp] = stg_tmp
#     # plot
#     plt.figure(1, figsize=(6, 6))
#     plt.plot(range(len(ac_stg_all)), ac_stg_all.iloc[:, 0], 'red')
#     if i < 9:
#         bench1 = w.wsd(code_bench[tmp], "close", sd, "2017-10-10", "Period=Q")
#         plt.plot(range(len(ac_stg_all)), bench1, 'blue')
#         bench2 = w.wsd(code_bench[tmp], "close", sd, "2017-10-10", "Period=Q")
#         plt.plot(range(len(ac_stg_all)), bench2, 'blueviolet')
#     plt.title(tmp)
