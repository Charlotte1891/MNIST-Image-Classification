from sklearn.preprocessing import StandardScaler
import util
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import decomposition
from sklearn.manifold import TSNE

# load dataset
x_train, y_train = util.load_dataset(training_mode=True)
m = x_train.shape[0]
n = x_train.shape[1]
x_train = x_train.reshape(m, n * n)

# standardize data
standardized_data = StandardScaler().fit_transform(x_train)

sample_data = standardized_data

# covariance matrix
covar_matrix = np.matmul(sample_data.T , sample_data)

# generates only the top 2 eigenvalues
values, vectors = eigh(covar_matrix, eigvals=(782,783))

vectors = vectors.T

new_coordinates = np.vstack((np.matmul(vectors, sample_data.T), y_train)).T

pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)
pca_data = np.vstack((pca_data.T, y_train)).T

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.savefig('PCA.png')
plt.show()


# PCA for dimensionality redcution (non-visualization)
pca = decomposition.PCA()
pca.n_components = 784
pca_data = pca.fit_transform(sample_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.savefig('PCA_dim.png')
plt.show()


# t-SNE
# subset of 10k size
data_10000 = standardized_data[0:10000,:]
labels_10000 = y_train[0:10000]

model = TSNE(n_components=2, random_state=0)

tsne_data = model.fit_transform(data_10000)

tsne_data = np.vstack((tsne_data.T, labels_10000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.savefig('t-SNE.png')
plt.show()
