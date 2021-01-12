import matplotlib.pyplot as plt
from dolfin import plot as dolfin_plot

def plot(function, filename):
    p = dolfin_plot(function)
    try:
        plt.colorbar(p)
    except AttributeError:
        pass
    plt.savefig(filename)
    plt.clf()
