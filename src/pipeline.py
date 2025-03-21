import pandas as pd
import osmnx as ox
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import time
from tqdm import tqdm
import numpy as np
from functools import partial
import pickle
from collections import defaultdict

# Global cache to avoid redundant nearest node calculations
# Using dict subclass for thread safety and better performance
class NodeCache(dict):
    def __missing__(self, key):
        return None

nearest_node_cache = NodeCache()
road_data_cache = {}  # Cache for road data

def load_data(file_path):
    """Load data from CSV file with optimized reading"""
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        parent_dir = os.path.dirname(file_path)
        if os.path.exists(parent_dir):
            print(f"Directory {parent_dir} exists. Contents:")
            print(os.listdir(parent_dir))
        else:
            print(f"Directory {parent_dir} does not exist!")
        return None
    
    # Use chunking to improve memory usage for large files
    # Only read necessary columns to reduce memory footprint
    required_columns = ['Order No', 'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long']
    
    # Check file size to determine chunking strategy
    file_size = os.path.getsize(file_path)
    chunk_size = 100000 if file_size > 100*1024*1024 else None  # Chunk if > 100MB
    
    if chunk_size:
        chunks = []
        for chunk in pd.read_csv(file_path, usecols=required_columns, chunksize=chunk_size):
            chunks.append(chunk)
        return pd.concat(chunks)
    else:
        return pd.read_csv(file_path, usecols=required_columns)

def load_network(counties, cache_file=None):
    """Load road network with improved caching strategy"""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading network from cache: {cache_file}")
        try:
            # Use pickle instead of networkx's read_gpickle for better performance
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached network: {e}. Rebuilding...")
    
    print(f"Loading road network for: {', '.join(counties)}")
    start_time = time.time()
    
    # Configure OSMNX for performance
    ox.config(use_cache=True, log_console=True)
    
    # Process counties in parallel if there are multiple
    if len(counties) > 1:
        with ThreadPoolExecutor() as executor:
            graphs = list(executor.map(
                lambda county: ox.graph_from_place(county, network_type='drive', simplify=True),
                counties
            ))
        
        # Combine all graphs
        G_combined = nx.compose_all(graphs)
    else:
        G_combined = ox.graph_from_place(counties[0], network_type='drive', simplify=True)
    
    # Precompute graph projections for faster distance calculations
    G_combined = ox.project_graph(G_combined)
    
    # Extract and cache common road attributes to reduce redundant lookups
    precompute_road_attributes(G_combined)
    
    # Save to cache if specified
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(G_combined, f, protocol=4)  # Protocol 4 for better performance with large objects
    
    elapsed = time.time() - start_time
    print(f"Road network loaded successfully in {elapsed:.2f} seconds")
    return G_combined

def precompute_road_attributes(G):
    """Precompute and cache common road attributes for faster lookups"""
    global road_data_cache
    
    print("Precomputing road attributes...")
    for u, v, data in tqdm(G.edges(data=True), desc="Caching road attributes"):
        # Cache relevant road attributes by edge
        edge_key = (u, v)
        road_data_cache[edge_key] = {
            'surface': data.get('surface', None),
            'tracktype': data.get('tracktype', None),
            'highway': data.get('highway', None)
        }
    
    print(f"Cached attributes for {len(road_data_cache)} road segments")

def get_nearest_node(lat, lon, G):
    """Get nearest node with improved caching"""
    global nearest_node_cache
    
    # Round to reduce cache size, improve hits
    cache_key = (round(lat, 6), round(lon, 6))
    
    # Check cache first
    cached_node = nearest_node_cache.get(cache_key)
    if cached_node is not None:
        return cached_node
    
    # Not in cache, calculate and store
    node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    nearest_node_cache[cache_key] = node
    return node

def get_road_quality(pickup_node, G):
    """Extract road quality with improved caching and lookup"""
    # Check connected edges for the pickup node
    edges = list(G.edges(pickup_node, data=True))
    if not edges:
        return 'unknown', None
    
    # Get first edge data
    edge_data = edges[0][2]
    
    # Extract road quality information
    road_quality = 'unknown'
    road_type = None
    
    # Try surface first
    surface = edge_data.get('surface')
    if surface:
        road_quality = surface
    else:
        # Try tracktype
        tracktype = edge_data.get('tracktype')
        if tracktype:
            road_quality = tracktype
        else:
            # Use highway type to infer quality
            highway = edge_data.get('highway')
            road_type = highway
            
            quality_mapping = {
                'motorway': 'paved', 'trunk': 'paved', 'primary': 'paved', 'secondary': 'paved', 
                'tertiary': 'paved', 'residential': 'paved', 'service': 'paved', 'unclassified': 'unpaved', 
                'track': 'unpaved', 'path': 'unpaved', 'footway': 'unpaved'
            }
            road_quality = quality_mapping.get(highway, 'unknown') if highway else 'unknown'
    
    return road_quality, road_type

def process_location_batch(batch_df, G):
    """Process a batch of locations efficiently"""
    results = []
    
    # Prepare arrays for faster bulk operations
    pickup_lats = batch_df['Pickup Lat'].values
    pickup_longs = batch_df['Pickup Long'].values
    dest_lats = batch_df['Destination Lat'].values
    dest_longs = batch_df['Destination Long'].values
    order_nos = batch_df['Order No'].values
    
    # Pre-calculate nearest nodes for all points in batch
    pickup_nodes = []
    dest_nodes = []
    
    for i in range(len(batch_df)):
        try:
            pickup_node = get_nearest_node(pickup_lats[i], pickup_longs[i], G)
            dest_node = get_nearest_node(dest_lats[i], dest_longs[i], G)
            
            pickup_nodes.append(pickup_node)
            dest_nodes.append(dest_node)
        except Exception as e:
            print(f"Error finding nodes for Order No {order_nos[i]}: {e}")
            pickup_nodes.append(None)
            dest_nodes.append(None)
    
    # Calculate road quality and shortest paths
    for i in range(len(batch_df)):
        pickup_node = pickup_nodes[i]
        dest_node = dest_nodes[i]
        
        if pickup_node is None or dest_node is None:
            results.append({
                'Order No': order_nos[i],
                'Road Quality': 'unknown',
                'Pickup Road Type': None,
                'Shortest Path Distance': None
            })
            continue
        
        try:
            # Get road quality and type
            road_quality, road_type = get_road_quality(pickup_node, G)
            
            # Calculate shortest path distance
            try:
                # Try with A* algorithm for better performance on large networks
                shortest_path = nx.astar_path_length(G, pickup_node, dest_node, weight='length',
                                                    heuristic=lambda u, v: ox.distance.great_circle(
                                                        G.nodes[u]['y'], G.nodes[u]['x'],
                                                        G.nodes[v]['y'], G.nodes[v]['x']
                                                    ))
            except:
                # Fall back to standard algorithm
                shortest_path = nx.shortest_path_length(G, pickup_node, dest_node, weight='length')
                
            results.append({
                'Order No': order_nos[i],
                'Road Quality': road_quality,
                'Pickup Road Type': road_type,
                'Shortest Path Distance': shortest_path
            })
        except Exception as e:
            print(f"Error processing data for Order No {order_nos[i]}: {e}")
            results.append({
                'Order No': order_nos[i],
                'Road Quality': 'unknown',
                'Pickup Road Type': None,
                'Shortest Path Distance': None
            })
    
    return results

def process_data_in_batches(data, G, dataset_name="dataset", batch_size=100):
    """Process data in batches with optimized parallelization"""
    total_rows = len(data)
    print(f"Processing {dataset_name} with {total_rows} records...")
    
    # Determine optimal batch size based on data size
    adjusted_batch_size = min(max(100, total_rows // 50), 1000)
    print(f"Using batch size: {adjusted_batch_size}")
    
    # Create batches for processing
    batches = [data.iloc[i:i+adjusted_batch_size] for i in range(0, total_rows, adjusted_batch_size)]
    
    # Set up multiprocessing (more efficient for CPU-bound tasks)
    # Use fewer workers than CPUs to avoid network contention
    num_workers = max(1, os.cpu_count() // 2)
    print(f"Using {num_workers} worker processes")
    
    # Process batches in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use partial to fix G parameter
        process_batch = partial(process_location_batch, G=G)
        
        # Map and process results with progress tracking
        for batch_results in tqdm(executor.map(process_batch, batches), 
                                 total=len(batches), 
                                 desc=f"Processing {dataset_name}"):
            all_results.extend(batch_results)
    
    # Convert results to DataFrame and merge with original data
    results_df = pd.DataFrame(all_results)
    
    # Merge back to original data on Order No
    output_data = data.merge(results_df, on='Order No', how='left')
    
    print(f"Processing of {dataset_name} complete!")
    return output_data

def full_data_pipeline(train_path, test_path, counties, output_dir=None, network_cache=None):
    """
    Execute the complete data processing pipeline with optimizations
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        counties: List of counties to build the road network from
        output_dir: Directory to save processed data (if None, won't save)
        network_cache: File path to cache network data (if None, won't cache)
    
    Returns:
        tuple: (processed_train_data, processed_test_data)
    """
    start_time = time.time()
    
    # Load datasets with optimized function
    print("Loading training data...")
    train_data = load_data(train_path)
    if train_data is None:
        print("Failed to load training data. Exiting.")
        return None, None
    print(f"Training data loaded: {len(train_data)} records")
    
    print("Loading test data...")
    test_data = load_data(test_path)
    if test_data is None:
        print("Failed to load test data. Exiting.")
        return train_data, None
    print(f"Test data loaded: {len(test_data)} records")
    
    # Load and prepare network with optimizations
    G = load_network(counties, cache_file=network_cache)
    
    # Process both datasets with improved batch processing
    processed_train = process_data_in_batches(train_data, G, "training data")
    processed_test = process_data_in_batches(test_data, G, "test data")
    
    # Save processed data if output directory is provided
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        train_basename = os.path.basename(train_path)
        test_basename = os.path.basename(test_path)
        train_output_path = os.path.join(output_dir, f"processed_{train_basename}")
        test_output_path = os.path.join(output_dir, f"processed_{test_basename}")
        
        print(f"Saving processed training data to {train_output_path}")
        processed_train.to_csv(train_output_path, index=False)
        
        print(f"Saving processed test data to {test_output_path}")
        processed_test.to_csv(test_output_path, index=False)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    # Return the processed data
    return processed_train, processed_test

if __name__ == "__main__":
    # Add debug info
    print(f"Current working directory: {os.getcwd()}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    project_root = os.path.dirname(script_dir)
    print(f"Project root: {project_root}")
    
    # Set paths
    datasets_folder = os.path.join(project_root, "datasets")
    train_file = os.path.join(datasets_folder, "TrainData.csv")
    test_file = os.path.join(datasets_folder, "TestData.csv")
    
    output_folder = os.path.join(project_root, "processed_data")
    network_cache_file = os.path.join(project_root, "cache", "network_cache.gpickle")
    
    counties = ["Nairobi, Kenya", "Kiambu, Kenya", "Machakos, Kenya"]
    
    # Process data with optimizations
    train_processed, test_processed = full_data_pipeline(
        train_file, 
        test_file, 
        counties, 
        output_folder, 
        network_cache = network_cache_file
    )
    
    # Display sample of processed data if available
    if train_processed is not None:
        print("\nSample of processed training data:")
        print(train_processed.head())
    
    if test_processed is not None:
        print("\nSample of processed test data:")
        print(test_processed.head())