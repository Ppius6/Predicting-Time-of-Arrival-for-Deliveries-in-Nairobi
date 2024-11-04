from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import osmnx as ox

# Calculate shortest path distance between pickup and destination points
def calculate_shortest_path_parallel(row, G):
    try:
        pickup_node = ox.distance.nearest_nodes(G, X=row['Pickup Long'], Y=row['Pickup Lat'])
        dest_node = ox.distance.nearest_nodes(G, X=row['Destination Long'], Y=row['Destination Lat'])
        shortest_path_distance = nx.shortest_path_length(G, pickup_node, dest_node, weight='length')
        return shortest_path_distance
    except Exception as e:
        print(f"Error processing row {row['Order No']}: {e}")
        return None

# Add shortest path distances to the data
def add_shortest_path_distances(data, G):
    with ThreadPoolExecutor() as executor:
        data['Shortest Path Distance'] = list(executor.map(lambda row: calculate_shortest_path_parallel(row, G), 
                                                           [row for _, row in data.iterrows()]))
    return data