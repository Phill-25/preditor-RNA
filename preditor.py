from yahooquery import Ticker
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot
import numpy as np
import pandas as pd
netflix = Ticker('nflx', asynchronous=True)
# petro = Ticker('PETR4.SA', asynchronous=True)
DSnetflix = netflix.history(period='max', interval='1d')
# DSnetflix.to_csv('out.csv')
def  media_movel_exp(periodo, DS):
    # MME = [ Valor Atual – Média Anterior ] x [ 2 / ( 1 + Número de Períodos ) ] + Média Anterior

    index_list = DS.index
    MME_list = []
    heder = DS.loc[index_list[0]:index_list[periodo-1]]
    tail = DS.loc[index_list[periodo]:]
    media_inicial = heder['volume'].sum()/periodo

    for n in range(periodo):
        MME_list.append(media_inicial)
    for idx, row in tail.iterrows():
        mme = (row['volume'] - MME_list[-1]) * ( 2/(1+periodo) )+ MME_list[-1]
        MME_list.append(mme)

    # s = pd.Series(MME_list, DS.index)
    DS.loc[:,'mme'] = MME_list
    # print([len(index_list), len(MME_list), DS])
    return DS

# calculo de 6 meses
begin = date(2021, 2, 1)
hoje = date.today()
fim = hoje+relativedelta(months=-6)
end = begin+relativedelta(months=-6)
# print(DSnetflix.loc[('nflx', date.today())])

# data-set de 6 meses a partir da data de hoje
# six_months_DS = DSnetflix.loc[('nflx', fim):('nflx', hoje)]

# for d, r in six_months_DS.iterrows():
#     print([d[0], r['open'], r['close'], r['volume'], r['high'], r['low'] ])
    # print([d, r])

# A func abaixo faz o calculo da média móvel para o DS
# e add essa coluna no mesmo. Portanto esse DS é a base do projeo
DB = media_movel_exp(21, DSnetflix)

DB.to_csv('with_mme.csv')
print(DB['open'])
D = []
for s, d in DB.index:
    D.append(d.__str__())
matplotlib.pyplot.plot(D, DB['volume'])
matplotlib.pyplot.plot(D, DB['mme'])
matplotlib.pyplot.show()
