from yahooquery import Ticker
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot
import numpy as np
import pandas as pd

from pybrain.datasets  import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer

from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader



netflix = Ticker('nflx', asynchronous=True)
# petro = Ticker('PETR4.SA', asynchronous=True)
print('Iniciando....')
print('Carregando os ativos...')

DSnetflix = netflix.history(period='max', interval='1d')

DSnetflix.to_csv('base_final.csv')

# open|volume|high|low|close|mme
""""
melhores resultados 
neuronios x camadas 
200 x 5 
200 x 3 => 36
300 x 3 => 36
400 x 3 => 32.
400 x 2 => 34
"""
# neuronios_nbr = 200
neuronios_nbr = 400

print('Montando a rede neural...')
preditor = buildNetwork(5,
                        neuronios_nbr, neuronios_nbr, neuronios_nbr,
                        1)

# preditor = NetworkReader.readFrom('')

print(preditor)


def media_movel_exp(periodo, DS):
    print('Calculando a média movel exponencial...')

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
    print('Limpando a base de dados...')

    return ds.drop(ds[ds.target == "nah"].index)


def carregar_o_futuro(dias_a_frente, ds):
    print('Adicionando os dados futuros...')

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

# open|volume|high|low|close|mme
def aprendendo_o_futuro(ds, epocas):
    print('Iniciando base de treino...')

    base = SupervisedDataSet(5, 1)

    for idx, row in ds.iterrows():
        base.addSample((row['open'], row['high'], row['low'],
                        row['close'], row['mme'] ), (row['target'],))

    treinador = BackpropTrainer(preditor, dataset=base, learningrate=0.01, momentum=0.06, weightdecay=0.01)
    print('Iniciando treinamento...')
    for i in range(1, epocas):
        erro = treinador.train()
        if i % 1 == 0:
            print(f"Erro da epoca {i}: {erro}")
            NetworkWriter.writeToFile(preditor, 'preditor_400x3_final.xml')
            # treinador.testOnData(dataset=base, verbose=False)

    NetworkWriter.writeToFile(preditor, 'preditor_400x3_final.xml')


def prevendo_o_futuro(ds):
    print('Prevendo o futuro...')
    preditos = []
    diffs = []
    for idx, row in ds.iterrows():
        predicao = preditor.activate([row['open'], row['high'], row['low'], row['close'], row['mme']])[0]
        preditos.append(predicao)

        diff = abs( row['target'] - predicao )
        diffs.append(diff)

    ds.loc[:, 'preditos'] = preditos
    ds.loc[:, 'diferenca'] = diffs
    ds.to_csv('preditos_final.csv')

    return ds


# A func abaixo faz o calculo da média móvel para o DS
# e add essa coluna no mesmo. Portanto esse DS é a base do projeo
DB = media_movel_exp(21, DSnetflix)

# a função abaixo cria uma nova coluna no dataset com o valor de fechamento do ativo para o periodo informado
DB = carregar_o_futuro(2, DB)


tamanho_db = DB.__len__()

index_list_bd = DB.index
limite_treino = round((tamanho_db/3) * 2)

base_treino = DB.loc[index_list_bd[0]:index_list_bd[limite_treino]]


aprendendo_o_futuro(base_treino, 1000)

base_teste = DB.loc[index_list_bd[++limite_treino]:index_list_bd[tamanho_db]]

prevendo_o_futuro(base_treino)

# print(DB['target'])
# DB.to_csv('with_taget.csv')

# D = []
# for s, d in DB.index:
#     D.append(d.__str__())
# matplotlib.pyplot.plot(D, DB['close'])
# # matplotlib.pyplot.plot(D, DB['mme'])
# matplotlib.pyplot.plot(D, DB['target'])
# matplotlib.pyplot.legend(['close',  'close - target'])
# matplotlib.pyplot.show()
