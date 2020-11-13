import matplotlib.pyplot as plt
from dolfin import plot as dolfin_plot

def plot(function, filename):
    p = dolfin_plot(function)
    plt.colorbar(p)
    plt.savefig(filename)
    plt.clf()
