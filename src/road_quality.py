import osmnx as ox
from concurrent.futures import ThreadPoolExecutor

nearest_node_cache_quality = {}

# Get road quality for a given location
def get_road_quality(lat, lon, G, order_no):
    try:
        if (lat, lon) not in nearest_node_cache_quality:
            nearest_node_cache_quality[(lat, lon)] = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        nearest_node = nearest_node_cache_quality[(lat, lon)]
        road_data = G[nearest_node]
        
        road_surfaces = [data[0].get('surface') for data in road_data.values()]
        if road_surfaces and road_surfaces[0]:
            return road_surfaces[0] if isinstance(road_surfaces[0], str) else road_surfaces[0][0]
        
        tracktypes = [data[0].get('tracktype') for data in road_data.values()]
        if tracktypes and tracktypes[0]:
            return tracktypes[0] if isinstance(tracktypes[0], str) else tracktypes[0][0]
        
        highway_types = [data[0].get('highway') for data in road_data.values()]
        quality_mapping = {
            'motorway': 'paved', 'trunk': 'paved', 'primary': 'paved', 'secondary': 'paved', 
            'tertiary': 'paved', 'residential': 'paved', 'service': 'paved', 'unclassified': 'unpaved', 
            'track': 'unpaved', 'path': 'unpaved', 'footway': 'unpaved'
        }
        return quality_mapping.get(highway_types[0], 'unknown') if highway_types else 'unknown'
    except Exception as e:
        print(f"Error getting road quality for location {lat}, {lon} in Order No {order_no}: {e}")
        return 'unknown'

# Add road qualities to the data
def add_road_qualities(data, G):
    with ThreadPoolExecutor() as executor:
        data['Road Quality'] = list(executor.map(lambda row: get_road_quality(row['Pickup Lat'], row['Pickup Long'], G, row['Order No']), 
                                                 [row for _, row in data.iterrows()]))
    return data
