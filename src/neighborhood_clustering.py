import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.neighbors import BallTree
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import pickle
import os
from tqdm import tqdm

# For parallel processing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def load_data(train_path='processed_data/processed_TrainData.csv', test_path='processed_data/processed_TestData.csv'):
    """Load the training and test datasets"""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def cluster_locations(df, eps=0.5, min_samples=5, method='dbscan'):
    """
    Cluster pickup and dropoff locations into neighborhoods
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing location data
    eps : float
        The maximum distance between two samples for them to be in the same cluster (DBSCAN)
    min_samples : int
        The number of samples in a neighborhood for a point to be considered a core point (DBSCAN)
    method : str
        Clustering method ('dbscan' or 'kmeans')
        
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with additional columns for pickup and destination clusters
    pickup_clusters : dict
        Dictionary with cluster information for pickups
    destination_clusters : dict
        Dictionary with cluster information for destinations
    """
    # Extract coordinates
    pickup_coords = df[['Pickup Lat', 'Pickup Long']].values
    dest_coords = df[['Destination Lat', 'Destination Long']].values
    
    # Create clusters
    if method == 'dbscan':
        pickup_clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pickup_coords)
        dest_clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(dest_coords)
        
        pickup_labels = pickup_clustering.labels_
        dest_labels = dest_clustering.labels_
    elif method == 'kmeans':
        # Determine optimal number of clusters using inertia plot (elbow method)
        n_clusters = 30  # Default, can be determined dynamically using elbow method
        
        pickup_clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pickup_coords)
        dest_clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(dest_coords)
        
        pickup_labels = pickup_clustering.labels_
        dest_labels = dest_clustering.labels_
    else:
        raise ValueError("Method must be 'dbscan' or 'kmeans'")
    
    # Add cluster labels to DataFrame
    df['Pickup_Cluster'] = pickup_labels
    df['Destination_Cluster'] = dest_labels
    
    # Create cluster information dictionaries
    pickup_clusters = {}
    for cluster_id in np.unique(pickup_labels):
        if cluster_id != -1:  # Skip noise points
            cluster_points = pickup_coords[pickup_labels == cluster_id]
            center = np.mean(cluster_points, axis=0)
            pickup_clusters[cluster_id] = {
                'center': center,
                'size': len(cluster_points),
                'lat_range': (np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])),
                'lng_range': (np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1]))
            }
    
    destination_clusters = {}
    for cluster_id in np.unique(dest_labels):
        if cluster_id != -1:  # Skip noise points
            cluster_points = dest_coords[dest_labels == cluster_id]
            center = np.mean(cluster_points, axis=0)
            destination_clusters[cluster_id] = {
                'center': center,
                'size': len(cluster_points),
                'lat_range': (np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])),
                'lng_range': (np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1]))
            }
    
    return df, pickup_clusters, destination_clusters

def calculate_neighborhood_statistics(df, pickup_clusters, destination_clusters, cache_file=None):
    """
    Calculate statistics for each neighborhood/cluster
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with cluster assignments
    pickup_clusters : dict
        Dictionary with cluster information for pickups
    destination_clusters : dict
        Dictionary with cluster information for destinations
    cache_file : str, optional
        Path to cache file to save/load results
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added neighborhood statistics
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading neighborhood statistics from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            neighborhood_stats = pickle.load(f)
        
        # Add neighborhood statistics to DataFrame
        for col, values in neighborhood_stats.items():
            df[col] = values
        
        return df
    
    print("Calculating neighborhood statistics...")
    
    # Calculate pickup cluster statistics
    pickup_stats = df.groupby('Pickup_Cluster').agg({
        'Time from Pickup to Arrival': ['mean', 'median', 'std', 'count'],
        'Distance (KM)': ['mean', 'median'],
        'Shortest Path Distance': ['mean', 'median'],
        'Order No': 'count'
    })
    
    # Calculate destination cluster statistics
    dest_stats = df.groupby('Destination_Cluster').agg({
        'Time from Pickup to Arrival': ['mean', 'median', 'std', 'count'],
        'Distance (KM)': ['mean', 'median'],
        'Shortest Path Distance': ['mean', 'median'],
        'Order No': 'count'
    })
    
    # Calculate within-cluster statistics
    df['Same_Cluster'] = df['Pickup_Cluster'] == df['Destination_Cluster']
    
    # Add cluster density (orders per area)
    for cluster_id, info in pickup_clusters.items():
        if cluster_id != -1:
            # Calculate approximate area in square kilometers
            lat_range = info['lat_range'][1] - info['lat_range'][0]
            lng_range = info['lng_range'][1] - info['lng_range'][0]
            # Rough approximation of area - for more accurate results, use proper geospatial tools
            area_approx = lat_range * lng_range * 111 * 111  # Roughly 111km per degree
            
            if area_approx > 0:
                pickup_clusters[cluster_id]['density'] = info['size'] / area_approx
            else:
                pickup_clusters[cluster_id]['density'] = np.nan
    
    # Map cluster statistics back to the dataframe
    df['Pickup_Cluster_Avg_Time'] = df['Pickup_Cluster'].map(
        {k: v for k, v in zip(pickup_stats.index, pickup_stats[('Time from Pickup to Arrival', 'mean')])}
    )
    
    df['Pickup_Cluster_Order_Count'] = df['Pickup_Cluster'].map(
        {k: v for k, v in zip(pickup_stats.index, pickup_stats[('Order No', 'count')])}
    )
    
    df['Pickup_Cluster_Density'] = df['Pickup_Cluster'].map(
        {k: info.get('density', np.nan) for k, info in pickup_clusters.items()}
    )
    
    df['Destination_Cluster_Avg_Time'] = df['Destination_Cluster'].map(
        {k: v for k, v in zip(dest_stats.index, dest_stats[('Time from Pickup to Arrival', 'mean')])}
    )
    
    df['Destination_Cluster_Order_Count'] = df['Destination_Cluster'].map(
        {k: v for k, v in zip(dest_stats.index, dest_stats[('Order No', 'count')])}
    )
    
    # Calculate pickup-to-destination cluster flow statistics
    flow_stats = df.groupby(['Pickup_Cluster', 'Destination_Cluster']).agg({
        'Time from Pickup to Arrival': ['mean', 'count'],
        'Distance (KM)': 'mean'
    }).reset_index()
    
    # Create a lookup dictionary for flow statistics
    flow_lookup = {}
    for _, row in flow_stats.iterrows():
        flow_lookup[(row['Pickup_Cluster'], row['Destination_Cluster'])] = {
            'avg_time': row[('Time from Pickup to Arrival', 'mean')],
            'count': row[('Time from Pickup to Arrival', 'count')],
            'avg_distance': row[('Distance (KM)', 'mean')]
        }
    
    # Map flow statistics back to the dataframe
    df['Cluster_Flow_Avg_Time'] = df.apply(
        lambda x: flow_lookup.get((x['Pickup_Cluster'], x['Destination_Cluster']), {}).get('avg_time', np.nan),
        axis=1
    )
    
    df['Cluster_Flow_Count'] = df.apply(
        lambda x: flow_lookup.get((x['Pickup_Cluster'], x['Destination_Cluster']), {}).get('count', 0),
        axis=1
    )
    
    # Classify neighborhoods as high/medium/low complexity based on average delivery time
    pickup_time_threshold_high = pickup_stats[('Time from Pickup to Arrival', 'mean')].quantile(0.75)
    pickup_time_threshold_low = pickup_stats[('Time from Pickup to Arrival', 'mean')].quantile(0.25)
    
    df['Pickup_Cluster_Complexity'] = df['Pickup_Cluster_Avg_Time'].apply(
        lambda x: 'High' if x > pickup_time_threshold_high else ('Low' if x < pickup_time_threshold_low else 'Medium')
    )
    
    # Normalize cluster statistics to create features
    df['Pickup_Cluster_Normalized_Time'] = (df['Pickup_Cluster_Avg_Time'] - 
                                           df['Pickup_Cluster_Avg_Time'].mean()) / df['Pickup_Cluster_Avg_Time'].std()
    
    df['Destination_Cluster_Normalized_Time'] = (df['Destination_Cluster_Avg_Time'] - 
                                                df['Destination_Cluster_Avg_Time'].mean()) / df['Destination_Cluster_Avg_Time'].std()
    
    # For reference: key statistics to judge cluster complexity
    pickup_cluster_complexity = {
        cluster_id: {
            'avg_time': stats[('Time from Pickup to Arrival', 'mean')],
            'complexity': 'High' if stats[('Time from Pickup to Arrival', 'mean')] > pickup_time_threshold_high else
                         ('Low' if stats[('Time from Pickup to Arrival', 'mean')] < pickup_time_threshold_low else 'Medium')
        }
        for cluster_id, stats in pickup_stats.iterrows()
    }
    
    # Cache the results if a cache file is provided
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Store the added columns
        neighborhood_stats = {
            'Pickup_Cluster_Avg_Time': df['Pickup_Cluster_Avg_Time'].values,
            'Pickup_Cluster_Order_Count': df['Pickup_Cluster_Order_Count'].values,
            'Pickup_Cluster_Density': df['Pickup_Cluster_Density'].values,
            'Destination_Cluster_Avg_Time': df['Destination_Cluster_Avg_Time'].values,
            'Destination_Cluster_Order_Count': df['Destination_Cluster_Order_Count'].values,
            'Cluster_Flow_Avg_Time': df['Cluster_Flow_Avg_Time'].values,
            'Cluster_Flow_Count': df['Cluster_Flow_Count'].values,
            'Pickup_Cluster_Complexity': df['Pickup_Cluster_Complexity'].values,
            'Pickup_Cluster_Normalized_Time': df['Pickup_Cluster_Normalized_Time'].values,
            'Destination_Cluster_Normalized_Time': df['Destination_Cluster_Normalized_Time'].values,
            'Same_Cluster': df['Same_Cluster'].values
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(neighborhood_stats, f)
    
    return df

def download_pois(city, poi_types=None, cache_file=None):
    """
    Download points of interest (POIs) for a city using OSMnx
    
    Parameters:
    -----------
    city : str
        Name of the city
    poi_types : list, optional
        List of POI types to download
    cache_file : str, optional
        Path to cache file to save/load results
    
    Returns:
    --------
    pois : geopandas GeoDataFrame
        GeoDataFrame containing POIs
    """
    if poi_types is None:
        poi_types = [
            'school', 'college', 'university',  # Educational
            'hospital', 'clinic', 'pharmacy',    # Healthcare
            'supermarket', 'market', 'mall', 'shop',  # Shopping
            'restaurant', 'cafe', 'bar', 'pub',  # Food & Drink
            'bank', 'atm', 'bureau_de_change',  # Financial
            'hotel', 'hostel', 'guest_house',  # Accommodation
            'bus_station', 'train_station', 'taxi_stand',  # Transport
            'park', 'garden',  # Leisure
            'police', 'fire_station', 'post_office',  # Public services
            'office', 'industrial',  # Business
            'stadium', 'theatre', 'cinema'  # Entertainment
        ]
    
    if cache_file and os.path.exists(cache_file):
        print(f"Loading POIs from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            pois = pickle.load(f)
        return pois
    
    print(f"Downloading POIs for {city}...")
    
    # Download POIs
    tags = {
        'amenity': poi_types,
        'shop': True,
        'leisure': True,
        'tourism': True,
        'office': True,
        'building': ['commercial', 'office', 'school', 'university', 'hospital', 'hotel']
    }
    
    try:
        pois = ox.geometries_from_place(city, tags=tags)
        
        # Clean and prepare the POI data
        if not pois.empty:
            # Add category column
            pois['category'] = ''
            
            for tag in ['amenity', 'shop', 'leisure', 'tourism', 'office', 'building']:
                if tag in pois.columns:
                    pois.loc[~pois[tag].isna(), 'category'] = pois.loc[~pois[tag].isna(), tag]
            
            # Extract lat/lon for easier use
            pois['lat'] = pois.geometry.centroid.y
            pois['lon'] = pois.geometry.centroid.x
            
            # Cache the results if a cache file is provided
            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(pois, f)
            
            return pois
        else:
            print(f"No POIs found for {city}")
            return None
    except Exception as e:
        print(f"Error downloading POIs for {city}: {e}")
        return None

def calculate_poi_distances(df, pois, cache_file=None):
    """
    Calculate distances to nearest POIs by category
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing location data
    pois : geopandas GeoDataFrame
        GeoDataFrame containing POIs
    cache_file : str, optional
        Path to cache file to save/load results
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added POI distance features
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading POI distances from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            poi_distances = pickle.load(f)
        
        # Add POI distances to DataFrame
        for col, values in poi_distances.items():
            df[col] = values
        
        return df
    
    if pois is None or pois.empty:
        print("No POIs available to calculate distances")
        return df
    
    print("Calculating distances to nearest POIs...")
    
    # Prepare BallTree for efficient nearest neighbor search
    poi_coords = np.radians(pois[['lat', 'lon']].values)
    tree = BallTree(poi_coords, metric='haversine')
    
    # Define key POI categories to analyze separately
    key_categories = {
        'school': ['school', 'college', 'university'],
        'healthcare': ['hospital', 'clinic', 'pharmacy'],
        'shopping': ['supermarket', 'market', 'mall', 'shop'],
        'food': ['restaurant', 'cafe', 'bar', 'pub'],
        'transport': ['bus_station', 'train_station', 'taxi_stand'],
        'office': ['office', 'commercial'],
        'leisure': ['park', 'garden', 'stadium', 'theatre', 'cinema']
    }
    
    # Function to find distance to nearest POI of a specific category
    def distance_to_nearest(lat, lon, category_filter):
        # Convert to radians
        coords = np.radians([[lat, lon]])
        
        # Filter POIs by category
        category_pois = pois[pois['category'].isin(category_filter)]
        
        if category_pois.empty:
            return np.nan
        
        category_coords = np.radians(category_pois[['lat', 'lon']].values)
        category_tree = BallTree(category_coords, metric='haversine')
        
        # Find nearest neighbor
        dist, _ = category_tree.query(coords, k=1)
        
        # Convert distance from radians to kilometers
        return dist[0][0] * 6371  # Earth radius in km
    
    # Batch process for efficiency
    pickup_coords = df[['Pickup Lat', 'Pickup Long']].values
    dest_coords = df[['Destination Lat', 'Destination Long']].values
    
    # Calculate distance to any POI (general)
    print("Calculating distances to nearest POIs (general)...")
    pickup_distances, _ = tree.query(np.radians(pickup_coords), k=1)
    dest_distances, _ = tree.query(np.radians(dest_coords), k=1)
    
    df['Pickup_Nearest_POI_Distance'] = pickup_distances.flatten() * 6371  # Convert to km
    df['Destination_Nearest_POI_Distance'] = dest_distances.flatten() * 6371  # Convert to km
    
    # Calculate distance to specific POI categories
    poi_distance_features = {}
    poi_distance_features['Pickup_Nearest_POI_Distance'] = df['Pickup_Nearest_POI_Distance'].values
    poi_distance_features['Destination_Nearest_POI_Distance'] = df['Destination_Nearest_POI_Distance'].values
    
    for category_name, category_list in tqdm(key_categories.items(), desc="Processing POI categories"):
        print(f"Calculating distances to nearest {category_name} POIs...")
        
        # Create feature name
        pickup_feature = f'Pickup_Nearest_{category_name}_Distance'
        dest_feature = f'Destination_Nearest_{category_name}_Distance'
        
        # Apply distance calculation for each row
        df[pickup_feature] = df.apply(
            lambda row: distance_to_nearest(row['Pickup Lat'], row['Pickup Long'], category_list),
            axis=1
        )
        
        df[dest_feature] = df.apply(
            lambda row: distance_to_nearest(row['Destination Lat'], row['Destination Long'], category_list),
            axis=1
        )
        
        # Store for caching
        poi_distance_features[pickup_feature] = df[pickup_feature].values
        poi_distance_features[dest_feature] = df[dest_feature].values
    
    # Calculate POI density features
    print("Calculating POI density features...")
    
    # Function to count POIs within a radius
    def count_pois_within_radius(lat, lon, radius_km=1.0):
        coords = np.radians([[lat, lon]])
        radius_rad = radius_km / 6371  # Convert km to radians
        
        # Count points within radius
        indices = tree.query_radius(coords, r=radius_rad)[0]
        return len(indices)
    
    # Apply POI density calculation
    df['Pickup_POI_Density_1km'] = df.apply(
        lambda row: count_pois_within_radius(row['Pickup Lat'], row['Pickup Long'], 1.0),
        axis=1
    )
    
    df['Destination_POI_Density_1km'] = df.apply(
        lambda row: count_pois_within_radius(row['Destination Lat'], row['Destination Long'], 1.0),
        axis=1
    )
    
    poi_distance_features['Pickup_POI_Density_1km'] = df['Pickup_POI_Density_1km'].values
    poi_distance_features['Destination_POI_Density_1km'] = df['Destination_POI_Density_1km'].values
    
    # Cache the results if a cache file is provided
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(poi_distance_features, f)
    
    return df

def identify_congestion_areas(df, threshold_percentile=75, cache_file=None):
    """
    Identify areas with high congestion based on delivery times
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing location and time data
    threshold_percentile : int
        Percentile threshold to define congestion (default: 75)
    cache_file : str, optional
        Path to cache file to save/load results
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added congestion area features
    congestion_areas : dict
        Dictionary with information about congestion areas
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading congestion areas from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            congestion_areas = cached_data['congestion_areas']
            congestion_features = cached_data['congestion_features']
        
        # Add congestion features to DataFrame
        for col, values in congestion_features.items():
            df[col] = values
        
        return df, congestion_areas
    
    print("Identifying congestion areas...")
    
    # Calculate speed for each delivery
    df['Speed_KMH'] = (df['Distance (KM)'] / (df['Time from Pickup to Arrival'] / 3600))
    
    # Identify threshold for congestion based on speed
    speed_threshold = np.percentile(df['Speed_KMH'].dropna(), 100 - threshold_percentile)
    
    print(f"Speed threshold for congestion: {speed_threshold:.2f} km/h")
    
    # Flag congested trips
    df['Is_Congested_Trip'] = df['Speed_KMH'] < speed_threshold
    
    # Group by pickup cluster to identify congested pickup areas
    pickup_congestion = df.groupby('Pickup_Cluster')['Is_Congested_Trip'].mean().reset_index()
    pickup_congestion.columns = ['Pickup_Cluster', 'Congestion_Ratio']
    
    # Define congestion threshold for clusters
    cluster_congestion_threshold = 0.5  # More than 50% of trips are congested
    
    # Identify congested clusters
    congested_clusters = pickup_congestion[pickup_congestion['Congestion_Ratio'] > cluster_congestion_threshold]['Pickup_Cluster'].tolist()
    
    print(f"Identified {len(congested_clusters)} congested clusters")
    
    # Build congestion areas information
    congestion_areas = {}
    for cluster_id in congested_clusters:
        cluster_data = df[df['Pickup_Cluster'] == cluster_id]
        
        if len(cluster_data) > 0:
            avg_lat = cluster_data['Pickup Lat'].mean()
            avg_long = cluster_data['Pickup Long'].mean()
            
            congestion_areas[cluster_id] = {
                'center': (avg_lat, avg_long),
                'congestion_ratio': pickup_congestion[pickup_congestion['Pickup_Cluster'] == cluster_id]['Congestion_Ratio'].values[0],
                'avg_speed': cluster_data['Speed_KMH'].mean(),
                'trip_count': len(cluster_data)
            }
    
    # Add features to the dataframe
    df['Pickup_In_Congested_Area'] = df['Pickup_Cluster'].isin(congested_clusters)
    df['Destination_In_Congested_Area'] = df['Destination_Cluster'].isin(congested_clusters)
    
    # Add congestion ratio for pickup and destination clusters
    congestion_ratio_dict = dict(zip(pickup_congestion['Pickup_Cluster'], pickup_congestion['Congestion_Ratio']))
    df['Pickup_Area_Congestion_Ratio'] = df['Pickup_Cluster'].map(congestion_ratio_dict)
    df['Destination_Area_Congestion_Ratio'] = df['Destination_Cluster'].map(congestion_ratio_dict)
    
    # Store congestion features for caching
    congestion_features = {
        'Speed_KMH': df['Speed_KMH'].values,
        'Is_Congested_Trip': df['Is_Congested_Trip'].values,
        'Pickup_In_Congested_Area': df['Pickup_In_Congested_Area'].values,
        'Destination_In_Congested_Area': df['Destination_In_Congested_Area'].values,
        'Pickup_Area_Congestion_Ratio': df['Pickup_Area_Congestion_Ratio'].values,
        'Destination_Area_Congestion_Ratio': df['Destination_Area_Congestion_Ratio'].values
    }
    
    # Cache the results if a cache file is provided
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'congestion_areas': congestion_areas,
                'congestion_features': congestion_features
            }, f)
    
    return df, congestion_areas

def visualize_clusters(df, pickup_clusters, congestion_areas=None, filepath='cluster_map.html'):
    """
    Create a folium map to visualize clusters and congestion areas
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing location data and cluster assignments
    pickup_clusters : dict
        Dictionary with cluster information
    congestion_areas : dict, optional
        Dictionary with congestion area information
    filepath : str
        Path to save the HTML map
    """
    # Create base map centered on Nairobi
    nairobi_coords = [-1.286389, 36.817223]  # Latitude, Longitude
    m = folium.Map(location=nairobi_coords, zoom_start=12)
    
    # Add cluster centers
    for cluster_id, info in pickup_clusters.items():
        if cluster_id != -1:  # Skip noise points
            center = info['center']
            size = info['size']
            popup_text = f"Cluster: {cluster_id}<br>Orders: {size}"
            
            # Determine if this is a congestion area
            is_congested = congestion_areas is not None and cluster_id in congestion_areas
            
            if is_congested:
                congestion_info = congestion_areas[cluster_id]
                popup_text += f"<br>Congestion Ratio: {congestion_info['congestion_ratio']:.2f}"
                popup_text += f"<br>Avg Speed: {congestion_info['avg_speed']:.2f} km/h"
                
                # Use red for congested areas
                folium.CircleMarker(
                    location=[center[0], center[1]],
                    radius=5 + size/200,  # Size based on number of orders
                    popup=popup_text,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6
                ).add_to(m)
            else:
                # Use blue for normal areas
                folium.CircleMarker(
                    location=[center[0], center[1]],
                    radius=5 + size/200,
                    popup=popup_text,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6
                ).add_to(m)
    
    # Sample a subset of orders for visualization (plotting all would be too dense)
    sample_df = df.sample(min(1000, len(df)))
    
    # Create a cluster marker for order points
    pickup_points = MarkerCluster(name="Pickup Points").add_to(m)
    
    for _, row in sample_df.iterrows():
        popup_text = f"Order: {row['Order No']}<br>Distance: {row['Distance (KM)']:.2f} km<br>Time: {row['Time from Pickup to Arrival']:.0f} s"
        
        folium.Marker(
            location=[row['Pickup Lat'], row['Pickup Long']],
            popup=popup_text,
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(pickup_points)
    
    # Add heatmap of pickup locations
    pickup_data = df[['Pickup Lat', 'Pickup Long']].values.tolist()
    HeatMap(pickup_data, radius=10, blur=15).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(filepath)
    print(f"Map saved to {filepath}")

def extract_geographic_features(train_data, test_data, output_dir='processed_data', cache_dir='cache'):
    """
    Main function to extract geographic features from both train and test datasets
    
    Parameters:
    -----------
    train_data : pandas DataFrame
        Training dataset
    test_data : pandas DataFrame
        Test dataset
    output_dir : str
        Directory to save processed data
    cache_dir : str
        Directory to save/load cache files
    
    Returns:
    --------
    train_data : pandas DataFrame
        Processed training dataset with geographic features
    test_data : pandas DataFrame
        Processed test dataset with geographic features
    """
    # Create cache and output directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cluster locations
    print("Clustering pickup and dropoff locations...")
    train_data, pickup_clusters, destination_clusters = cluster_locations(
        train_data, 
        eps=0.01,  # ~1km
        min_samples=5, 
        method='dbscan'
    )
    
    # Apply same clustering to test data
    def assign_cluster(lat, lon, clusters):
        min_dist = float('inf')
        assigned_cluster = -1
        
        for cluster_id, info in clusters.items():
            center = info['center']
            dist = np.sqrt((lat - center[0])**2 + (lon - center[1])**2)
            if dist < min_dist:
                min_dist = dist
                assigned_cluster = cluster_id
        
        return assigned_cluster
    
    test_data['Pickup_Cluster'] = test_data.apply(
        lambda row: assign_cluster(row['Pickup Lat'], row['Pickup Long'], pickup_clusters),
        axis=1
    )
    
    test_data['Destination_Cluster'] = test_data.apply(
        lambda row: assign_cluster(row['Destination Lat'], row['Destination Long'], destination_clusters),
        axis=1
    )
    
    # 2. Calculate neighborhood statistics
    print("Calculating neighborhood statistics...")
    train_data = calculate_neighborhood_statistics(
        train_data, 
        pickup_clusters, 
        destination_clusters,
        cache_file=os.path.join(cache_dir, 'neighborhood_stats.pkl')
    )
    
    # Apply neighborhood statistics to test data
    columns_to_copy = [
        'Pickup_Cluster_Avg_Time', 'Pickup_Cluster_Order_Count', 'Pickup_Cluster_Density',
        'Destination_Cluster_Avg_Time', 'Destination_Cluster_Order_Count',
        'Cluster_Flow_Avg_Time', 'Cluster_Flow_Count',
        'Pickup_Cluster_Complexity', 'Pickup_Cluster_Normalized_Time',
        'Destination_Cluster_Normalized_Time'
    ]
    
    for col in columns_to_copy:
        if col in train_data.columns:
            # Create mapping dictionaries
            if 'Pickup_Cluster' in col:
                mapping = train_data.groupby('Pickup_Cluster')[col].mean().to_dict()
                test_data[col] = test_data['Pickup_Cluster'].map(mapping)
            elif 'Destination_Cluster' in col:
                mapping = train_data.groupby('Destination_Cluster')[col].mean().to_dict()
                test_data[col] = test_data['Destination_Cluster'].map(mapping)
            elif 'Cluster_Flow' in col:
                # Create flow mapping
                flow_mapping = {}
                for pickup_cluster in test_data['Pickup_Cluster'].unique():
                    for dest_cluster in test_data['Destination_Cluster'].unique():
                        flow_data = train_data[(train_data['Pickup_Cluster'] == pickup_cluster) & 
                                              (train_data['Destination_Cluster'] == dest_cluster)]
                        if len(flow_data) > 0:
                            flow_mapping[(pickup_cluster, dest_cluster)] = flow_data[col].mean()
                
                # Apply flow mapping
                test_data[col] = test_data.apply(
                    lambda row: flow_mapping.get((row['Pickup_Cluster'], row['Destination_Cluster']), np.nan),
                    axis=1
                )
    
    test_data['Same_Cluster'] = test_data['Pickup_Cluster'] == test_data['Destination_Cluster']
    
    # 3. Download and process POIs
    print("Downloading and processing POIs...")
    pois = download_pois(
        city='Nairobi, Kenya',
        cache_file=os.path.join(cache_dir, 'nairobi_pois.pkl')
    )
    
    # 4. Calculate POI distances
    print("Calculating POI distances...")
    train_data = calculate_poi_distances(
        train_data, 
        pois,
        cache_file=os.path.join(cache_dir, 'train_poi_distances.pkl')
    )
    
    test_data = calculate_poi_distances(
        test_data, 
        pois,
        cache_file=os.path.join(cache_dir, 'test_poi_distances.pkl')
    )
    
    # 5. Identify congestion areas
    print("Identifying congestion areas...")
    train_data, congestion_areas = identify_congestion_areas(
        train_data,
        threshold_percentile=75,
        cache_file=os.path.join(cache_dir, 'congestion_areas.pkl')
    )
    
    # Apply congestion features to test data
    test_data['Pickup_In_Congested_Area'] = test_data['Pickup_Cluster'].isin(congestion_areas.keys())
    test_data['Destination_In_Congested_Area'] = test_data['Destination_Cluster'].isin(congestion_areas.keys())
    
    # Map congestion ratios from train to test
    congestion_ratio_dict = train_data.groupby('Pickup_Cluster')['Pickup_Area_Congestion_Ratio'].mean().to_dict()
    test_data['Pickup_Area_Congestion_Ratio'] = test_data['Pickup_Cluster'].map(congestion_ratio_dict)
    test_data['Destination_Area_Congestion_Ratio'] = test_data['Destination_Cluster'].map(congestion_ratio_dict)
    
    # 6. Visualize clusters
    print("Creating visualization...")
    visualize_clusters(
        train_data, 
        pickup_clusters,
        congestion_areas=congestion_areas,
        filepath=os.path.join(output_dir, 'cluster_map.html')
    )
    
    # 7. Save processed data
    print("Saving processed data...")
    train_data.to_csv(os.path.join(output_dir, 'geographic_features_train.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'geographic_features_test.csv'), index=False)
    
    return train_data, test_data

if __name__ == "__main__":
    # Load data
    train_data, test_data = load_data()
    
    # Extract geographic features
    train_data, test_data = extract_geographic_features(train_data, test_data)
    
    # Display sample of features
    print("\nSample of geographic features (train):")
    geographic_columns = [col for col in train_data.columns if 'Cluster' in col or 'POI' in col or 'Congestion' in col]
    print(train_data[geographic_columns].head())