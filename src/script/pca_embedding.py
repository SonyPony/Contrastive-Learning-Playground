import torch
import numpy as np
import matplotlib.pyplot as plt
from util.plot import set_style as plt_set_style
from sklearn.decomposition import PCA


plt_set_style()
batch_size = 128
data = torch.load(f"../outputs/embeddings_{batch_size}.pth")
embeddings, labels = data["embedding"].numpy(), data["label"].numpy()

mse = list()
components_list = [8, 16, 32, 64, 128, 256, 512]

components_list = [512,]
print(embeddings.shape)
for components in components_list:
    pca = PCA(n_components=components)
    pca.fit(embeddings)

    reduced = pca.transform(embeddings)
    reconstructed = pca.inverse_transform(reduced)
    #mse.append(((reconstructed - embeddings) ** 2).mean())
    a = np.std(reduced, axis=0)
    print(a.shape)
    a = a.max()
    mse.append(a)

plt.figure(figsize=(8,4))
plt.plot(components_list, mse)
plt.ylabel("Max std")
plt.xlabel("#PCA Components")
plt.show()