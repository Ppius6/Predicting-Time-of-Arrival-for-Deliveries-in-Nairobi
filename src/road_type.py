import osmnx as ox
from concurrent.futures import ThreadPoolExecutor

nearest_node_cache = {}

# Get road type for a given location
def get_road_type(lat, lon, G, order_no):
    try:
        if (lat, lon) not in nearest_node_cache:
            nearest_node_cache[(lat, lon)] = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        nearest_node = nearest_node_cache[(lat, lon)]
        road_data = G[nearest_node]
        road_types = [data[0].get('highway') for data in road_data.values()]
        return road_types[0] if isinstance(road_types[0], str) else road_types[0][0] if road_types else None
    except Exception as e:
        print(f"Error getting road type for location {lat}, {lon} in Order No {order_no}: {e}")
        return None

# Add road types to the data
def add_road_types(data, G):
    with ThreadPoolExecutor() as executor:
        data['Pickup Road Type'] = list(executor.map(lambda row: get_road_type(row['Pickup Lat'], row['Pickup Long'], G, row['Order No']), 
                                                     [row for _, row in data.iterrows()]))
    return data
