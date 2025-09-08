"""
Geolocation module for crop disease hotspot mapping
Provides functionality to track disease locations and create interactive maps
"""

import folium
import json
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests

class DiseaseLocationTracker:
    def __init__(self, db_path="data/disease_locations.db"):
        """
        Initialize disease location tracker
        
        Args:
            db_path (str): Path to SQLite database for storing location data
        """
        self.db_path = db_path
        self.init_database()
        self.geolocator = Nominatim(user_agent="crop_disease_detector")
        
    def init_database(self):
        """Initialize SQLite database for storing disease detection records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS disease_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                plant_type TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                location_name TEXT,
                user_id TEXT,
                image_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def record_detection(self, latitude, longitude, plant_type, disease_name, 
                        confidence, user_id=None, image_path=None):
        """
        Record a disease detection with location data
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            plant_type (str): Type of plant detected
            disease_name (str): Name of disease detected
            confidence (float): Confidence score of detection
            user_id (str): Optional user identifier
            image_path (str): Optional path to detection image
            
        Returns:
            int: Record ID of the inserted detection
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get location name from coordinates
        location_name = self.get_location_name(latitude, longitude)
        
        cursor.execute('''
            INSERT INTO disease_detections 
            (timestamp, latitude, longitude, plant_type, disease_name, 
             confidence, location_name, user_id, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            latitude,
            longitude,
            plant_type,
            disease_name,
            confidence,
            location_name,
            user_id,
            image_path
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Disease detection recorded with ID: {record_id}")
        return record_id
    
    def get_location_name(self, latitude, longitude):
        """
        Get human-readable location name from coordinates
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            str: Location name or coordinates if geocoding fails
        """
        try:
            location = self.geolocator.reverse(f"{latitude}, {longitude}")
            return location.address if location else f"{latitude}, {longitude}"
        except Exception as e:
            print(f"Geocoding error: {e}")
            return f"{latitude}, {longitude}"
    
    def get_detections_in_radius(self, center_lat, center_lng, radius_km):
        """
        Get all disease detections within a specified radius
        
        Args:
            center_lat (float): Center latitude
            center_lng (float): Center longitude
            radius_km (float): Radius in kilometers
            
        Returns:
            list: List of detection records within radius
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM disease_detections')
        all_detections = cursor.fetchall()
        conn.close()
        
        nearby_detections = []
        center_point = (center_lat, center_lng)
        
        for detection in all_detections:
            detection_point = (detection[2], detection[3])  # lat, lng
            distance = geodesic(center_point, detection_point).kilometers
            
            if distance <= radius_km:
                nearby_detections.append({
                    'id': detection[0],
                    'timestamp': detection[1],
                    'latitude': detection[2],
                    'longitude': detection[3],
                    'plant_type': detection[4],
                    'disease_name': detection[5],
                    'confidence': detection[6],
                    'location_name': detection[7],
                    'user_id': detection[8],
                    'image_path': detection[9],
                    'distance_km': round(distance, 2)
                })
        
        return nearby_detections
    
    def get_disease_hotspots(self, min_detections=3, radius_km=5):
        """
        Identify disease hotspots based on detection density
        
        Args:
            min_detections (int): Minimum number of detections to consider a hotspot
            radius_km (float): Radius to search for nearby detections
            
        Returns:
            list: List of hotspot locations with statistics
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM disease_detections', conn)
        conn.close()
        
        if df.empty:
            return []
        
        hotspots = []
        processed_locations = set()
        
        for _, detection in df.iterrows():
            location_key = f"{detection['latitude']:.3f},{detection['longitude']:.3f}"
            
            if location_key in processed_locations:
                continue
                
            nearby = self.get_detections_in_radius(
                detection['latitude'], 
                detection['longitude'], 
                radius_km
            )
            
            if len(nearby) >= min_detections:
                # Calculate hotspot statistics
                diseases = [d['disease_name'] for d in nearby]
                plants = [d['plant_type'] for d in nearby]
                
                disease_counts = {}
                plant_counts = {}
                
                for disease in diseases:
                    disease_counts[disease] = disease_counts.get(disease, 0) + 1
                    
                for plant in plants:
                    plant_counts[plant] = plant_counts.get(plant, 0) + 1
                
                most_common_disease = max(disease_counts, key=disease_counts.get)
                most_common_plant = max(plant_counts, key=plant_counts.get)
                
                hotspot = {
                    'center_lat': detection['latitude'],
                    'center_lng': detection['longitude'],
                    'location_name': detection['location_name'],
                    'total_detections': len(nearby),
                    'radius_km': radius_km,
                    'most_common_disease': most_common_disease,
                    'most_common_plant': most_common_plant,
                    'disease_counts': disease_counts,
                    'plant_counts': plant_counts,
                    'detections': nearby
                }
                
                hotspots.append(hotspot)
                
                # Mark all nearby locations as processed
                for det in nearby:
                    loc_key = f"{det['latitude']:.3f},{det['longitude']:.3f}"
                    processed_locations.add(loc_key)
        
        return hotspots

class DiseaseMapGenerator:
    def __init__(self, tracker):
        """
        Initialize map generator
        
        Args:
            tracker (DiseaseLocationTracker): Disease location tracker instance
        """
        self.tracker = tracker
        
    def create_disease_map(self, center_lat=None, center_lng=None, zoom_start=10,
                          save_path="static/disease_map.html"):
        """
        Create an interactive map showing disease detections and hotspots
        
        Args:
            center_lat (float): Map center latitude (auto-calculated if None)
            center_lng (float): Map center longitude (auto-calculated if None)
            zoom_start (int): Initial zoom level
            save_path (str): Path to save the HTML map
            
        Returns:
            folium.Map: Interactive map object
        """
        # Get all detections
        conn = sqlite3.connect(self.tracker.db_path)
        df = pd.read_sql_query('SELECT * FROM disease_detections', conn)
        conn.close()
        
        if df.empty:
            print("No disease detections found in database")
            return None
        
        # Calculate map center if not provided
        if center_lat is None or center_lng is None:
            center_lat = df['latitude'].mean()
            center_lng = df['longitude'].mean()
        
        # Create base map
        disease_map = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add individual detection markers
        self.add_detection_markers(disease_map, df)
        
        # Add hotspot markers
        hotspots = self.tracker.get_disease_hotspots()
        self.add_hotspot_markers(disease_map, hotspots)
        
        # Add legend
        self.add_map_legend(disease_map)
        
        # Save map
        disease_map.save(save_path)
        print(f"Disease map saved to: {save_path}")
        
        return disease_map
    
    def add_detection_markers(self, disease_map, detections_df):
        """Add individual detection markers to map"""
        
        # Color mapping for different disease types
        color_map = {
            'healthy': 'green',
            'blight': 'red',
            'rust': 'orange',
            'spot': 'purple',
            'scab': 'darkred',
            'rot': 'black',
            'mildew': 'blue',
            'mosaic': 'yellow'
        }
        
        for _, detection in detections_df.iterrows():
            # Determine marker color based on disease
            color = 'gray'  # default
            for disease_type, marker_color in color_map.items():
                if disease_type.lower() in detection['disease_name'].lower():
                    color = marker_color
                    break
            
            # Create popup content
            popup_content = f"""
            <b>Plant:</b> {detection['plant_type']}<br>
            <b>Disease:</b> {detection['disease_name']}<br>
            <b>Confidence:</b> {detection['confidence']:.2%}<br>
            <b>Location:</b> {detection['location_name']}<br>
            <b>Date:</b> {detection['timestamp'][:10]}<br>
            <b>Coordinates:</b> {detection['latitude']:.4f}, {detection['longitude']:.4f}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[detection['latitude'], detection['longitude']],
                radius=8,
                popup=folium.Popup(popup_content, max_width=300),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(disease_map)
    
    def add_hotspot_markers(self, disease_map, hotspots):
        """Add hotspot markers to map"""
        
        for hotspot in hotspots:
            # Create hotspot circle
            folium.Circle(
                location=[hotspot['center_lat'], hotspot['center_lng']],
                radius=hotspot['radius_km'] * 1000,  # Convert to meters
                popup=f"Hotspot: {hotspot['total_detections']} detections",
                color='red',
                fillColor='red',
                fillOpacity=0.2,
                weight=3
            ).add_to(disease_map)
            
            # Create detailed popup content
            popup_content = f"""
            <b>Disease Hotspot</b><br>
            <b>Location:</b> {hotspot['location_name']}<br>
            <b>Total Detections:</b> {hotspot['total_detections']}<br>
            <b>Most Common Disease:</b> {hotspot['most_common_disease']}<br>
            <b>Most Common Plant:</b> {hotspot['most_common_plant']}<br>
            <b>Radius:</b> {hotspot['radius_km']} km<br>
            """
            
            # Add hotspot center marker
            folium.Marker(
                location=[hotspot['center_lat'], hotspot['center_lng']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='warning-sign')
            ).add_to(disease_map)
    
    def add_map_legend(self, disease_map):
        """Add legend to the map"""
        
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 180px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Disease Map Legend</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Healthy</p>
        <p><i class="fa fa-circle" style="color:red"></i> Blight</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Rust</p>
        <p><i class="fa fa-circle" style="color:purple"></i> Spot</p>
        <p><i class="fa fa-circle" style="color:blue"></i> Mildew</p>
        <p><i class="fa fa-circle" style="color:black"></i> Rot</p>
        <p>🔴 Disease Hotspot</p>
        </div>
        '''
        
        disease_map.get_root().html.add_child(folium.Element(legend_html))

def get_user_location():
    """
    Get user's current location using IP-based geolocation
    
    Returns:
        tuple: (latitude, longitude) or (None, None) if failed
    """
    try:
        response = requests.get('http://ipapi.co/json/', timeout=5)
        data = response.json()
        return data.get('latitude'), data.get('longitude')
    except Exception as e:
        print(f"Could not get location: {e}")
        return None, None

def main():
    """Main function for testing geolocation features"""
    print("Disease Geolocation System Test")
    
    # Initialize tracker
    tracker = DiseaseLocationTracker()
    
    # Add some sample data for testing
    sample_detections = [
        (40.7128, -74.0060, "Tomato", "Late_blight", 0.85),  # New York
        (40.7589, -73.9851, "Tomato", "Early_blight", 0.78),  # Manhattan
        (40.6892, -74.0445, "Potato", "Late_blight", 0.92),  # Jersey City
        (40.7282, -73.7949, "Corn", "Northern_Leaf_Blight", 0.88),  # Queens
    ]
    
    print("Adding sample detections...")
    for lat, lng, plant, disease, conf in sample_detections:
        tracker.record_detection(lat, lng, plant, disease, conf, user_id="test_user")
    
    # Generate map
    map_generator = DiseaseMapGenerator(tracker)
    disease_map = map_generator.create_disease_map()
    
    if disease_map:
        print("Disease map created successfully!")
        print("Open static/disease_map.html in a web browser to view the map")
    
    # Get hotspots
    hotspots = tracker.get_disease_hotspots(min_detections=2)
    print(f"\nFound {len(hotspots)} disease hotspots")
    
    for i, hotspot in enumerate(hotspots, 1):
        print(f"\nHotspot {i}:")
        print(f"  Location: {hotspot['location_name']}")
        print(f"  Detections: {hotspot['total_detections']}")
        print(f"  Most common disease: {hotspot['most_common_disease']}")

if __name__ == "__main__":
    main()