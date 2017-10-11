import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
from pyamg.strength import classical_strength_of_connection as strength

def plot():
    # Create Matrices (A, S) for plotting
    eps = 0.001
    theta = np.pi / 8.0
    n = 10
    sten = diffusion_stencil_2d(epsilon=eps, theta=theta)
    A = stencil_grid(sten, (n, n)).tocsr()
    S = strength(A)
    G = nx.from_scipy_sparse_matrix(A)
    GS = nx.from_scipy_sparse_matrix(S)
    pos = dict()
    for i in range(n):
        for j in range(n):
            pos[i*n+j] = (i,j)


    # Plot Serial CF Splitting
    c_nodes = list()
    f_nodes = list()
    split_file = open("../../build/raptor/ruge_stuben/tests/aniso_splitting.txt")
    node = 0
    for line in split_file:
        state = (int)(line.rsplit('\n')[0])
        if state == 1:
            c_nodes.append(node)
        else:
            f_nodes.append(node)
        node += 1
    split_file.close()

    nx.draw_networkx_nodes(G, pos, c_nodes, node_color='blue', linewidths=None)
    nx.draw_networkx_nodes(G, pos, f_nodes, node_color='white', linewidths=None)
    nx.draw_networkx_edges(G, pos, style = "dashed", width=0.3)
    nx.draw_networkx_edges(GS, pos, width=1.0)

    plt.xlim([-0.5, n-0.5])
    plt.ylim([-0.5, n-0.5])
    plt.savefig("aniso_splitting.pdf")
    plt.clf()

    # Plot CF Splitting on 2 Processes
    def plot_parallel(num_procs):
        node_procs = list()
        c_nodes = list()
        f_nodes = list()
        node = 0
        for i in range(num_procs):
            split_file = open("../../build/raptor/ruge_stuben/tests/aniso_splitting_%d_%d.txt"
                    %(num_procs, i))
            for line in split_file:
                state = (int)(line.rsplit('\n')[0])
                if state == 1:
                    c_nodes.append(node)
                else:
                    f_nodes.append(node)
                node_procs.append(i)
                node += 1
            split_file.close()
        on_proc_list = []
        boundary_list = []
        for edge in G.edges():
            if (node_procs[edge[0]] != node_procs[edge[1]]):
                boundary_list.append(edge)
            else:
                on_proc_list.append(edge)

        nx.draw_networkx_nodes(G, pos, c_nodes, node_color='blue', linewidths=None)
        nx.draw_networkx_nodes(G, pos, f_nodes, node_color='white', linewidths=None)
        nx.draw_networkx_edges(GS, pos, style = "dashed", edgelist = boundary_list,
                width = 0.3)
        nx.draw_networkx_edges(GS, pos, width=1.0, edgelist = on_proc_list)

        plt.xlim([-0.5, n-0.5])
        plt.ylim([-0.5, n-0.5])
        plt.savefig("aniso_splitting_np_%d.pdf" %num_procs)
        plt.clf()

    plot_parallel(2)
    plot_parallel(4)
    plot_parallel(8)
    plot_parallel(16)

if __name__=='__main__':
    plot()

