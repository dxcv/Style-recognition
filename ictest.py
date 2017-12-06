
factor = ['abs_ret1', 'abs_ret3', 'beta', 'calmar1', 'calmar3', 'down1', 'down3', 'loss1', 'loss3', 'mdd1', 'mdd3', 'sharp1', 'sharp3', 'stock', 'timing',
          'vol1', 'vol3']
for i in os.listdir('Data2'):
    for j in os.listdir('Data2/' + i):
        ic_tmp = np.zeros([len(factor), len(first_day) - 1])
        for d in range(len(first_day) - 1):
            fd = first_day[d]
            ld = last_day[d]
            if os.path.exists('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv'):
                x = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv')
                y = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'Y.csv')
                for k in range(len(factor)):
                    tmp_fct = factor[k]
                    a1 = np.array(x[tmp_fct])
                    a2 = np.array(y['Y'])
                    a1[np.isnan(a2)] = 0
                    a2[np.isnan(a2)] = 0
                    ic_tmp[k,d] = np.corrcoef(a1, a2)[0][1]
        ic_df = pd.DataFrame(ic_tmp, index=factor, columns=first_day[0:-1])
        ic_df.to_csv('Data2/' + i + j + 'ictest.csv')



factor = ['abs_ret1', 'abs_ret2', 'beta', 'calmar1', 'calmar3', 'down1', 'down3', 'loss1', 'loss3', 'mdd1', 'mdd3', 'sharp1', 'sharp3', 'stock', 'timing',
          'vol1', 'vol3']
for i in os.listdir('Data2'):
    for j in os.listdir('Data2/' + i):
        ic_tmp = np.zeros([len(factor), len(first_day) - 1])
        for d in range(len(first_day) - 1):
            fd = first_day[d]
            ld = last_day[d]
            if os.path.exists('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv'):
                x = pd.read_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'X.csv')
                if isinstance(x['Unnamed: 0'][0], str):
                    code = x['Unnamed: 0']
                else:
                    code = x['Unnamed: 0.1']
                yfd = w.wsd(",".join(code), "NAV_acc", str(fd), str(fd), "")
                yld = w.wsd(",".join(code), "NAV_acc", str(ld), str(ld), "")
                logret = pd.DataFrame({'code': code, 'Y': np.log(np.array(yld.Data[0]) / np.array(yfd.Data[0]))})
                logret.to_csv('Data2/' + i + '/' + j + '/' + str(fd) + 'Y.csv')
                for k in range(len(factor)):
                    tmp_fct = factor[k]
                    a1 = np.array(x[tmp_fct])
                    a2 = np.array(logret['Y'])
                    ic_tmp[k,d] = np.corrcoef(a1, a2)[0][1]
        ic_df = pd.DataFrame(ic_tmp, index=factor, columns=first_day[0:-1])
        ic_df.to_csv('Data2/' + i + '/' + j + '/' + 'ictest.csv')






for i in os.listdir('Data2'):
    for j in os.listdir('Data2/'+i):
        for d in range(len(first_day)):
            fd = first_day[d]
            # ld = last_day[d]
            if os.path.exists('Data2/'+i+ '/' + j + '/' + str(fd) + 'X.csv'):
                x = pd.read_csv('Data2/'+i+ '/' + j + '/' + str(fd) + 'X.csv')
                if ('asset' in list(x.columns)) & ('operating_period' in list(x.columns)):
                    del x['asset']
                    del x['operating_period']
                    for k in range(len(x)):
                        if str(k) in list(x['Unnamed: 0']):
                            break
                    x1 = x.iloc[0:list(x['Unnamed: 0']).index(str(k)),:]
                    x2 = x1.drop(x1[np.sum(x1,axis=1)==0].index)
                    x2 = x2.reset_index()
                    x2.to_csv('Data2/'+i+ '/' + j + '/' + str(fd) + 'X.csv')

for i in os.listdir('Data2'):
    for j in os.listdir('Data2/'+i):
        for d in range(len(first_day)):
            fd = first_day[d]
            ld = last_day[d]
            x = pd.read_csv('Data2/'+i+ '/' + j + '/' + str(fd) + 'X.csv')
            if ('asset' in list(x.columns)) & ('operating_period' in list(x.columns)):
                del x['asset']
                del x['operating_period']
                for k in range(len(x)):
                    if str(k) in list(x['Unnamed: 0']):
                        break
                x1 = x.iloc[0:list(x['Unnamed: 0']).index(str(k)),:]
                x2 = x1.drop(x1[np.sum(x1,axis=1)==0].index)
                x2 = x2.reset_index()
                x2.to_csv('Data2/'+i+ '/' + j + '/' + str(fd) + 'X.csv')


for i in fundtype1:
    for j in fundtype2:
        for d in range(len(first_day)):
            fd = first_day[i]
            ld = last_day[i]
            x = pd.read_csv('Data2/'+i+ '/' + j + '/' + str(fd) + 'X.csv')
            code = x['Unnamed: 0']
            yfd = w.wsd(",".join(code), "NAV_acc", str(fd), str(fd), "")
            yld = w.wsd(",".join(code), "NAV_acc", str(ld), str(ld), "")
            logret = pd.DataFrame({'code':code,'Y':np.log(np.array(yld.Data[0])/np.array(yfd.Data[0]))})
            logret.to_csv('Data2/'+i+ '/' + j + '/' + str(fd) + 'Y.csv')
            factor = x.columns[1:]
            for i in factor:
                a1 = np.array(x[i])
                a2 = np.array(logret['Y'])
                ic = np.corrcoef(a1, a2)[0][1]
