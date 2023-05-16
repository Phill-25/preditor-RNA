from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.datasets  import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

# para salvar uma rede
# NetworkWriter.writeToFile(net, 'filename.xml')
# net = NetworkReader.readFrom('filename.xml')

rede = buildNetwork(2, 3, 1)
#
# rede = FeedForwardNetwork()
#
# #numero de neuronios de entrada
# camada_entrada = LinearLayer(2)
#
# #numero de neuronios da camada oculta
# camada_oculta = SigmoidLayer(3)
#
# #numero de neuronios de saida
# camada_saida = SigmoidLayer(1)
#
#
# # definição dos pesos
# bias1 = BiasUnit()
# bias2 = BiasUnit()
#
# # estrutura da rede neural
# rede.addModule(camada_entrada)
# rede.addModule(camada_oculta)
# rede.addModule(camada_saida)
# rede.addModule(bias1)
# rede.addModule(bias2)
#
# # ligação entre as camadas
# conec_entrada_to_oculta = FullConnection(camada_entrada, camada_oculta)
# conec_oculta_to_saida = FullConnection(camada_oculta, camada_saida)
# # adição dos pesos
# conec_bias_to_oculta = FullConnection(bias1, camada_oculta)
# conec_bias_to_saida = FullConnection(bias2, camada_saida)
#
# # constroi a rede
# rede.sortModules()
print('Ativação da rede antes do treinamento')
print('0 0 %s' % round( rede.activate([0,0])[0] ) )
print('0 1 %s' % round( rede.activate([0,1])[0] ) )
print('1 0 %s' % round( rede.activate([1,0])[0] ) )
print('1 1 %s' % round( rede.activate([1,1])[0] ) )

base = SupervisedDataSet(2, 1)

base.addSample((0, 0), (0,))
base.addSample((0, 1), (1,))
base.addSample((1, 0), (1,))
base.addSample((1, 1), (0,))

print('Iniciando treinamento...')
treinador =  BackpropTrainer(rede, dataset=base, learningrate=0.01, momentum=0.06)

for i in range(1, 10000):
    erro = treinador.train()
    if i % 1000 == 0:
        print('Erro: %s' % erro)

print('Ativando RNA treinada...')
print('0 0 %s' % round( rede.activate([0,0])[0] ) )
print('0 1 %s' % round( rede.activate([0,1])[0] ) )
print('1 0 %s' % round( rede.activate([1,0])[0] ) )
print('1 1 %s' % round( rede.activate([1,1])[0] ) )