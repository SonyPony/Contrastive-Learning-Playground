import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from util.plot import set_style


set_style()
def colors():
    return ['#14C8B7', '#EDB732', '#AD70FF', '#FF617B', '#FF8A4F', '#0053D6']

batch_sizes = ["random", 32, 64, 128, 256, 512, 1024]
for batch_size in batch_sizes:
    data = torch.load(f"../outputs/embeddings_{batch_size}.pth")
    embeddings, labels = data["embedding"].numpy(), data["label"].numpy()

    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    z = tsne.fit_transform(embeddings)


    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=colors()[:5], #sns.color_palette("hls", 5),
                    legend=False,
                    alpha=0.3,
                    data=df).set(title=f"{batch_size} T-SNE projection")
    plt.xlim([-70, 70])
    plt.ylim([-70, 70])
    plt.show()