import matplotlib.pyplot as plt
import matplotlib as mpl

def set_style():
    title_color = "223991"
    axes_color = 'C3C8D1'
    axes_textcolor = '6C6C77'

    plt.rcParams["font.family"] = "DM Mono"
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=['14C8B7', 'EDB732', 'AD70FF', 'FF617B', 'FF8A4F', '0053D6'])
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.titlesize"] = 25
    plt.rcParams["axes.titlecolor"] = title_color

    plt.rcParams["axes.labelsize"] = 22 #16
    plt.rcParams["axes.labelpad"] = 8
    plt.rcParams["axes.labelcolor"] = title_color

    plt.rcParams["axes.edgecolor"] = axes_color

    for axis in ("x", "y"):
        plt.rcParams[f"{axis}tick.major.size"] = 5
        plt.rcParams[f"{axis}tick.major.width"] = 1.5
        plt.rcParams[f"{axis}tick.major.pad"] = 6.5

        plt.rcParams[f"{axis}tick.color"] = axes_color
        plt.rcParams[f"{axis}tick.labelcolor"] = axes_textcolor
        plt.rcParams[f"{axis}tick.labelsize"] = 17#12

    plt.rcParams["legend.fontsize"] = 17
    plt.rcParams["legend.labelcolor"] = title_color
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "upper center"

    plt.rcParams["figure.subplot.top"] = 0.90
    plt.rcParams["figure.subplot.bottom"] = 0.23
    plt.rcParams["figure.subplot.right"] = 0.98
    plt.rcParams["figure.subplot.left"] = 0.15