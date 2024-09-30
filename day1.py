

#class Personagem:
   # def __init__(self, idade, nome):
        #self.idade = idade
        #self.nome = nome
        
    #def personagem1(self):
   #         self.idade += 1

# = Personagem(25, "Eduardo")

#personagem.personagem1()

#print(personagem.idade)

import numpy as np        # Para operações numéricas
import pandas as pd       # Para manipulação de dados
import matplotlib.pyplot as plt  # Para visualização de dados
import seaborn as sns     # Para visualizações estatísticas
from sklearn.model_selection import train_test_split  # Para dividir dados
from sklearn.linear_model import LinearRegression      # Exemplo de modelo
from sklearn.metrics import mean_squared_error 


data = {
    'nome': [ 'ana', 'bruno', 'miguel'],
    'idade': [12, 13, 10],
    'salario': [2000, 3000, 4000]
    
}

#df virou a 'data'
df = pd.DataFrame(data)

#VISUALIZAÇÃO DE DADOS, mostra as 5 primeiras linhas


print(df.head())

print(df.describe())  # Estatísticas resumidas

alta_renda = df[df['salario'] > 3000]

#Gráfico de Dispersão

plt.scatter(df['idade'], df['salario'])
plt.title('idade vs salário')
plt.xlabel('idade')
plt.ylabel('salário')
plt.show()

#Histograma

plt.hist(df['Idade'], bins=5)
plt.title('Distribuição de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

#MODELO DE MACHINE LEARNING COM SCIKIT-LEARN

X = df[['Idade']]  # Feature
y = df['Salario']  # Target

#atribuição de variáveis de treino e variáveis de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression() #ativando o modelo de regressão linear

modelo.fit(X_train, y_train) #treinando o modelo, melhor se ajusta aos dados

previsoes = modelo.predict(X_test) #é o modelo que aprendeu a relação do código anterior,
#como se o treinamento estivesse sendo representado por 'x_test'.

print(previsoes)


#mse é um metódo utilizado para avaliar a precisão do modelo DE REGRESSÃO 
mse = mean_squared_error(y_test, previsoes)
#mean_squared_error vem pronta da bilioteca e recebe o valor de 'y_test e previsões'
#o mse tem uma fórmula PRÓPRIA 
print(f'Mean Squared Error: {mse}')






