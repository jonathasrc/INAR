import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
pd.set_option('display.notebook_repr_html', False)
plt.style.use('seaborn-white')
df = pd.read_csv('datasets/USArrests.csv', index_col=0)
print("info")
print(df.info())
print("Mean")
print(df.mean())
print("var")
print(df.var())
X = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
pca_loadings = pd.DataFrame(PCA().fit(X).components_.T, index=df.columns, columns=['V1', 'V2', 'V3', 'V4'])
print(pca_loadings)
pca = PCA()
df_plot = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3', 'PC4'],
                       index=X.index)
print(df_plot)
fig, ax1 = plt.subplots(figsize=(9,7))
ax1.set_xlim(-3.5,3.5)
ax1.set_ylim(-3.5,3.5)
for i in df_plot.index:
    ax1.annotate(i, (-df_plot.PC1.loc[i], -df_plot.PC2.loc[i]), ha='center')
ax1.hlines(0, -3.5,3.5, linestyles='dotted', colors='grey')
ax1.vlines(0, -3.5,3.5, linestyles='dotted', colors='grey')
ax1.set_xlabel('First_Principal_Component')
ax1.set_ylabel('Second_Principal_Component')
ax2 = ax1.twinx().twiny()
ax2.set_ylim(-1,1)
ax2.set_xlim(-1,1)
ax2.tick_params(axis='y', colors='orange')
ax2.set_xlabel('Principal_Component_loading_vectors',
                   color='orange')
a = 1.07
for i in pca_loadings[['V1', 'V2']].index:
    ax2.annotate(i, (-pca_loadings.V1.loc[i]*a, -pca_loadings.V2.loc[i]*a), color='orange')
ax2.arrow(0,0,-pca_loadings.V1[0], -pca_loadings.V2[0])
ax2.arrow(0,0,-pca_loadings.V1[1], -pca_loadings.V2[1])
ax2.arrow(0,0,-pca_loadings.V1[2], -pca_loadings.V2[2])
ax2.arrow(0,0,-pca_loadings.V1[3], -pca_loadings.V2[3])
plt.savefig('img/pcausscrimesplot.png')
plt.show()