from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()

#numero de neuronios de entrada
camada_entrada = LinearLayer(2)

#numero de neuronios da camada oculta
camada_oculta = SigmoidLayer(3)

#numero de neuronios de saida
camada_saida = SigmoidLayer(1)


# definição dos pesos
bias1 = BiasUnit()
bias2 = BiasUnit()

# estrutura da rede neural
rede.addModule(camada_entrada)
rede.addModule(camada_oculta)
rede.addModule(camada_saida)
rede.addModule(bias1)
rede.addModule(bias2)

# ligação entre as camadas
conec_entrada_to_oculta = FullConnection(camada_entrada, camada_oculta)
conec_oculta_to_saida = FullConnection(camada_oculta, camada_saida)
# adição dos pesos
conec_bias_to_oculta = FullConnection(bias1, camada_oculta)
conec_bias_to_saida = FullConnection(bias2, camada_saida)

# constroi a rede
rede.sortModules()

print(rede)

print(conec_entrada_to_oculta.params)
print(conec_oculta_to_saida.params)

print(conec_bias_to_oculta.params)
print(conec_bias_to_saida.params)