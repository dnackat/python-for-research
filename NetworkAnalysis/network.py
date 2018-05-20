#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 20:18:01 2018

@author: dileepn
"""

import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

G = nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.savefig("karate_graph.pdf")

def er_graph(N, p):
    """ Generate an Erdos-Renyi graph. """
    # Create empty graph
    G = nx.Graph()
    # Add all N nodes in the graph
    G.add_nodes_from(range(N))
    # Loop over all pairs of nodes and add an edge with probability, p
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p = p):
                G.add_edge(node1, node2)
                
    return G
                
nx.draw(er_graph(50, 0.08), node_size=40, \
        with_labels=True, node_color="gray")
plt.savefig("er1.pdf")

def plot_degree_distribution(G):
    degree_sequence = [G.degree(d) for d in G.nodes()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")
    
G1 = er_graph(500, 0.08)
plot_degree_distribution(G1)
G2 = er_graph(500, 0.08)
plot_degree_distribution(G2)
G3 = er_graph(500, 0.08)
plot_degree_distribution(G3)
plt.savefig("hist3.pdf")

# Social network analysis of two villages in India
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=",")
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=",")

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    degree_sequence = [G.degree(d) for d in G.nodes()]
    print("Average degree: %.2f" % np.mean(degree_sequence))
  
plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig("village_hist.pdf")

# gen = nx.connected_component_subgraphs(G1), g = gen.__next__() --> iterator
# object: returns the next component

# Largest connected componenent (LCC)
G1_LCC = max(nx.connected_component_subgraphs(G1), key=len)
G2_LCC = max(nx.connected_component_subgraphs(G2), key=len)

print(G1_LCC.number_of_nodes() / G1.number_of_nodes())
print(G2_LCC.number_of_nodes() / G2.number_of_nodes())

plt.figure()
nx.draw(G1_LCC, node_color="red", edge_color="gray", node_size=20)
plt.savefig("village1.pdf")

nx.draw(G2_LCC, node_color="green", edge_color="gray", node_size=20)
plt.savefig("village2.pdf")

