import torch
import matplotlib.pyplot as plt
from util.plot import set_style, colors
from tqdm import tqdm


set_style()
batch_size_list = (32, 64, 128, 256, 512, 1024)


def load_data(positive: bool):
    whole_data = {}

    for batch_size in batch_size_list:
        if positive:
            FILE = f"../outputs/data_{batch_size}.pth"
        else:
            FILE = f"../outputs/data_{batch_size}_neg.pth"

        complete_data = torch.load(FILE)

        total_count = 0
        total_elems_sum = 0
        total_elems_diff_sum = 0
        batch_data = []

        for class_count, data in complete_data.items():
            data = torch.tensor(data)
            batch_data.append(data)

            count = data.shape[0]
            elems_sum = torch.sum(data)
            mean = elems_sum / count
            elems_diff_sum = torch.sum((data - mean) ** 2)

            total_elems_sum += elems_sum
            total_count += count
            total_elems_diff_sum += elems_diff_sum

            #print(f"Class ID: {class_count}")
            #print(f"Std: {(elems_diff_sum / count) ** 0.5}, Mean: {mean}")

        batch_data = torch.cat(batch_data)
        whole_data[batch_size] = batch_data
        print(f"Batch: {batch_size}, Total Std: {(total_elems_diff_sum / total_count) ** 0.5}, Total Mean: {total_elems_sum / total_count}")

    # plotting
    return torch.squeeze(torch.dstack(list(whole_data.values()))).T



# plot:
fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(5)

x = list(range(2, 13, 2))

for positive in (False, True):
    D = load_data(positive=positive)
    vp = ax.violinplot(D, x, widths=1,
                       showmeans=True, showmedians=False, showextrema=True)
    # styling:
    color = "red" if not positive else "#6C6C77"
    for i, body in enumerate(vp['bodies']):
        body.set_alpha(0.1)
        #if i != 2:
        #    body.set_facecolor("#C3C8D1")
        body.set_facecolor(color)
        if not positive:
            body.set_alpha(0.3)


    p = vp["cmeans"]
    cmean_colors = p.get_color()
    colors = [
        color
        #"#C3C8D1" if i != 128 else cmean_colors[0]
        for i in batch_size_list
    ]

    p.set_color(colors)
    p.set_linewidth(2)

    for k in ["cmaxes", "cmins", "cbars"]:
        p = vp[k]
        p.set_linewidth(2)
        p.set_alpha(0.6)

        cmean_colors = p.get_color()
        p.set_color(colors)


ax.set_xticks(x)
ax.set_xticklabels(batch_size_list)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Similarity")
plt.show()
