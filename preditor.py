from yahooquery import Ticker
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot
import numpy as np
import pandas as pd


netflix = Ticker('nflx', asynchronous=True)
# petro = Ticker('PETR4.SA', asynchronous=True)
DSnetflix = netflix.history(period='max', interval='1d')


def media_movel_exp(periodo, DS):
    # MME = [ Valor Atual – Média Anterior ] x [ 2 / ( 1 + Número de Períodos ) ] + Média Anterior
    index_list = DS.index
    MME_list = []
    heder = DS.loc[index_list[0]:index_list[periodo-1]]
    tail = DS.loc[index_list[periodo]:]
    media_inicial = heder['close'].sum()/periodo

    for n in range(periodo):
        MME_list.append(media_inicial)
    for idx, row in tail.iterrows():
        mme = (row['close'] - MME_list[-1]) * (2/(1+periodo)) + MME_list[-1]
        MME_list.append(mme)

    # s = pd.Series(MME_list, DS.index)
    DS.loc[:, 'mme'] = MME_list
    # print([len(index_list), len(MME_list), DS])
    return DS


def limpa_data_set(ds):
    return ds.drop(ds[ds.target == "nah"].index)


def carregar_o_futuro(dias_a_frente, ds):

    lista_do_futuro = []
    for idx, row in ds.iterrows():

        ativo_dia = row.name[1]
        dia_futuro = ativo_dia+relativedelta(days=dias_a_frente)

        try:
            ativo_futuro = ds.loc[('nflx', dia_futuro):('nflx', dia_futuro)]
            valor_fechamento_futuro = ativo_futuro['close'][0]
            lista_do_futuro.append(valor_fechamento_futuro)
        except:
            lista_do_futuro.append("nah")

    ds.loc[:, 'target'] = lista_do_futuro
    return limpa_data_set(ds)





# A func abaixo faz o calculo da média móvel para o DS
# e add essa coluna no mesmo. Portanto esse DS é a base do projeo
DB = media_movel_exp(21, DSnetflix)


# a função abaixo cria uma nova coluna no dataset com o valor de fechamento do ativo para o periodo informado
DB = carregar_o_futuro(30, DB)
print(DB['target'])
DB.to_csv('with_taget.csv')

D = []
for s, d in DB.index:
    D.append(d.__str__())
matplotlib.pyplot.plot(D, DB['close'])
# matplotlib.pyplot.plot(D, DB['mme'])
matplotlib.pyplot.plot(D, DB['target'])
matplotlib.pyplot.legend(['close',  'close - target'])
matplotlib.pyplot.show()
