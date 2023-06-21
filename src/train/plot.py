import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

PATH = "logs/lightning_logs/version_1/"
FILE = "metrics.csv"


def plot_loss():
    metrics = pd.read_csv(f"{PATH}/{FILE}")
    del metrics["step"]
    del metrics["val_acc"]
    del metrics["val_f1"]
    del metrics["lr-Adam"]
    metrics.set_index("epoch", inplace=True)
    metrics.dropna(axis=1, how="all").head()
    sn.relplot(data=metrics, kind="line")
    plt.show()


def plot_accuracy():
    metrics = pd.read_csv(f"{PATH}/{FILE}")
    del metrics["step"]
    del metrics["train_loss"]
    del metrics["val_loss"]
    del metrics["lr-Adam"]
    del metrics["val_f1"]
    metrics.set_index("epoch", inplace=True)
    metrics.dropna(axis=1, how="all").head()
    sn.relplot(data=metrics, kind="line")
    plt.show()


def plot_f1():
    metrics = pd.read_csv(f"{PATH}/{FILE}")
    del metrics["step"]
    del metrics["train_loss"]
    del metrics["val_loss"]
    del metrics["lr-Adam"]
    del metrics["val_acc"]
    metrics.set_index("epoch", inplace=True)
    metrics.dropna(axis=1, how="all").head()
    sn.relplot(data=metrics, kind="line")
    plt.show()


def plot_acc_f1():
    metrics = pd.read_csv(f"{PATH}/{FILE}")
    del metrics["step"]
    del metrics["train_loss"]
    del metrics["val_loss"]
    del metrics["lr-Adam"]
    metrics.set_index("epoch", inplace=True)
    metrics.dropna(axis=1, how="all").head()
    sn.relplot(data=metrics, kind="line")
    plt.show()


plot_loss()

plot_accuracy()

plot_f1()

plot_acc_f1()
