import numpy as np
import matplotlib.pyplot as plt
from util.plot import set_style as plt_set_style


plt_set_style()
plt.figure(figsize=(8, 5))
embedding_sizes = np.array([64, 128, 256, 512])
accs = np.array([0.452, 0.464, 0.436, 0.444])

plt.plot(embedding_sizes, accs * 100)
plt.xlabel("Embedding Size")
plt.ylabel("Accuracy [%]")
plt.show()