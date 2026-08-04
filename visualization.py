import matplotlib.pyplot as plt

def plot_selected(axe, chi1, q1, target):
    axe.scatter(chi1, q1, c=target, marker=".")

def plot_res_diff(chi1, q1, target, res_fun):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(chi1, q1, c=target, marker=".")
    axes[1].scatter(chi1, q1, c=res_fun, marker=".")
