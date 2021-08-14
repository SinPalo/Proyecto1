# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:02:34 2021

@author: Sinuhe
"""

# Bajamos las librerias que utlizamos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import cluster
from sklearn.preprocessing import Normalizer
from sklearn.impute import KNNImputer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Leemos la ubicación del arhivo
path = os.getcwd()
os.chdir(path)

# Leemos los datos
raw_data = pd.read_csv('paises.csv',decimal=',')

# Tomamos los datos las columnas que seran utiles para el analisis
df = raw_data.iloc[:,[4,6,7,8,9,11,12,15,16,17,18,19]].copy()
nombres = ['densidad poblacional', 'migración neta', 'mortalidad infantil', 
           'PIB per capita', 'alfabetizacion', 'arable', 'cultivado', 'tasa natilidad', 'tasa mortalidad', 'agricultura',
          'industria', 'servicios']
df.columns = nombres

# Calculamos el porcentaje de datos faltantes
df.isnull().sum()/len(df)


# Hacemos la imputación de daltos faltantes
df = pd.DataFrame(KNNImputer(n_neighbors=2, weights = 'uniform').fit_transform(df))
df.columns = nombres


# Normalizamos los daots
df_normalizado = pd.DataFrame(Normalizer().fit_transform(df))


# Hacemos la visualización t-SNE sin considerar los clusters
tsne = TSNE(learning_rate= 50, random_state=123, perplexity=25).fit_transform(df_normalizado)
df_normalizado['comp_1'] = tsne[:,0]
df_normalizado['comp_2'] = tsne[:,1]

sns.scatterplot(x="comp_1", y="comp_2", data=df_normalizado).set(title="T-SNE proyección")
df_normalizado.drop('comp_1', inplace=True, axis=1)
df_normalizado.drop('comp_2', inplace=True, axis=1)


# Graficamos un heatmap para ver las correlaciones
sns.heatmap(df.corr(),annot = True)
# Graficamos un pairplot para visualizar la relación entre las variables
sns.pairplot(df)

# Hacemos el gráfico de PCA
#n_components toma los componentes necesarios para recuperar el 95% de varianza
pca = PCA(n_components=0.95)
PC = pca.fit_transform(df_normalizado)
pca_df = pd.DataFrame(PC)


inercia = []
# Hacemos la gráfica de inercia
for k in range(1,8):
    clf = cluster.KMeans(random_state=1, n_clusters = k)
    clf.fit(pca_df)
    inercia.append(np.sqrt(clf.inertia_))
    
plt.plot(range(1,8), inercia, marker='s')
plt.xlabel('núm. de clusters')
plt.ylabel('Inercia')

# Nos tomamos 4 clusters, salvamos esto en una variable
clf = cluster.KMeans(random_state=1, n_clusters = 4).fit(df)

# Graficamos el pairplot considerando los clusters
etiquetas = pd.DataFrame(clf.labels_)
df_etiquetado = pd.concat((df,etiquetas), axis=1)
df_etiquetado = df_etiquetado.rename({0: 'cluster'}, axis=1)
sns.pairplot(df_etiquetado,hue='cluster', palette = ["C0", "C1", "k","C3"])

# Graficamos el t-SNE considerando los clusters
df_etiquetado['comp_1'] = tsne[:,0]
df_etiquetado['comp_2'] = tsne[:,1]

sns.scatterplot(x="comp_1", y="comp_2", hue='cluster', palette = ["C0", "C1", "k","C3"],data=df_etiquetado).set(title="T-SNE proyección")

# Calculamos los países que perteneen a cada cluster
print('Del grupo 0 hay ' + str(sum(df_etiquetado['cluster']==0)) + ' países.')
print('Del grupo 1 hay ' + str(sum(df_etiquetado['cluster']==1)) + ' países.')
print('Del grupo 2 hay ' + str(sum(df_etiquetado['cluster']==2)) + ' países.')
print('Del grupo 3 hay ' + str(sum(df_etiquetado['cluster']==3)) + ' países.')

df_etiquetado.insert(0, 'Países', raw_data['Country'])

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Observamos los países que pertenecen a cada cluster
print(df_etiquetado[df_etiquetado['cluster']==0])
print(df_etiquetado[df_etiquetado['cluster']==1])
print(df_etiquetado[df_etiquetado['cluster']==2])
print(df_etiquetado[df_etiquetado['cluster']==3])

# Imprimos los centros
centros = pd.DataFrame(clf.cluster_centers_)
centros.columns = nombres
print(centros.sort_values(by=['PIB per capita']))































