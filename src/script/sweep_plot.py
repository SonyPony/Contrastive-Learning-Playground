import wandb
import numpy as np
import matplotlib.pyplot as plt

from util.plot import set_style, colors
from tqdm import tqdm


set_style()
api = wandb.Api()

class_count = [5, 10, 50, 100, 200]
data = {"sup": {}, "ss": {}}
for key in data.keys():
    for c in class_count:
        data[key][c] = {}

for run in tqdm(api.runs("sonypony/clp", filters={"$or": [{"tags": "sweep_sup"}, {"tags": "sweep_ss"}]})):
    name = " ".join(run.name.split("-")[-2:])
    assert ("sup" in run.description) or "ss" in run.description

    if run.config["experiment_cfg/data/dataset/num_classes"] < 100:
        continue

    exp_type, _, class_count, _, batch_size = run.description.split("\n")[-1].split(" ")
    class_count = int(class_count)
    batch_size = int(batch_size)

    accuracies = run.scan_history(keys=["val/acc-1"])
    accuracies = np.array([(row["val/acc-1"]) for row in accuracies])
    acc_1 = np.max(accuracies, axis=0)

    data[exp_type][class_count][batch_size] = acc_1

# plotting data
plt.figure(figsize=(12,5))

colors = colors()
for i, num_classes in enumerate((100, 200)):
    for training_type in ("sup", "ss"):
        plot_data = data[training_type][num_classes]
        linestyle = "--" if training_type == "sup" else "-"
        plt.plot(plot_data.keys(),  np.array(list(plot_data.values()))* 100, linestyle, linewidth=2, color=f"#{colors[i]}")
plt.ylabel("Accuracy [%]")
plt.xlabel("Batch size")
plt.savefig("num_class.pdf")
plt.show()

print(data)
exit(0)

keys = np.array(list(data.keys()))
indices = np.argsort(keys)

le_keys = np.array(list(le_data.keys()))
le_indices = np.argsort(le_keys)

plt.figure(figsize=(12,5))
plt.plot(keys[indices], (np.array(list(data.values()))[indices])[..., 0] * 100, linewidth=4)
plt.plot(le_keys[le_indices], (np.array(list(le_data.values()))[le_indices])[..., 0] * 100, linewidth=4)
plt.xlabel("False Negative Rate [%]")
plt.grid(linestyle='--', color='#C3C8D1')
plt.ylabel("Top-1 Acc. [%]")
leg = plt.legend(["Memory Bank", "Linear Protocol"], ncol=2, bbox_to_anchor=(0.5, 1.15))
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)

plt.savefig("acc.pdf")
plt.show()