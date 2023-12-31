import matplotlib.pyplot as plt


def setup_pyplot():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.style.use("seaborn-v0_8-muted")

    plt.rcParams["figure.figsize"] = (14, 8)
    # plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plt.rcParams["axes.formatter.limits"] = -3, 3
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["axes.formatter.useoffset"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "gainsboro"

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
