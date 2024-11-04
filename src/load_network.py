import osmnx as ox
import networkx as nx

# Load road network for the region (Nairobi, Kiambu, Machakos areas)
def load_network(counties):
    G_combined = None
    for county in counties:
        G = ox.graph_from_place(county, network_type = 'drive')
        G_combined = G if G_combined is None else nx.compose(G_combined, G)
    return G_combined
