import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.load_network import load_network
from src.shortest_path import add_shortest_path_distances
from src.road_type import add_road_types
from src.road_quality import add_road_qualities
from src.load_data import load_and_preprocess_data

# Define counties and load the road network
counties = ['Nairobi, Kenya', 'Kiambu, Kenya', 'Machakos, Kenya']
G_combined = load_network(counties)

# Load and preprocess data
TrainData = load_and_preprocess_data('datasets/TrainData.csv')
TestData = load_and_preprocess_data('datasets/TestData.csv')

# Add features
TrainData = add_shortest_path_distances(TrainData, G_combined)
TestData = add_shortest_path_distances(TestData, G_combined)
TrainData = add_road_types(TrainData, G_combined)
TestData = add_road_types(TestData, G_combined)
TrainData = add_road_qualities(TrainData, G_combined)
TestData = add_road_qualities(TestData, G_combined)

# Save consolidated datasets
TrainData.to_csv('datasets/TrainDataEnriched.csv', index = False)
TestData.to_csv('datasets/TestDataEnriched.csv', index = False)