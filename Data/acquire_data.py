import ee
import osmnx as ox
import geopandas as gpd

# 1. Initialize Google Earth Engine
# This will prompt you to log into your Google Cloud account in the terminal
ee.Authenticate()
ee.Initialize(project='spring-radar-478010-k1')

# 2. Define Bengaluru Ward 1 Bounding Box 
# (Replace with the exact coordinates for your specific pilot ward)
north,south,east,west = 13.00, 12.97, 77.62, 77.58
ward_bbox = ee.Geometry.BBox(west, south, east, north)

# Set timeframe to peak summer months to capture high heat stress
start_date = '2025-03-01'
end_date = '2025-05-31'

# 3. Pull GEE Datasets
print("Querying GEE Datasets...")

# Landsat 8 (Surface Temperature)
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(ward_bbox) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUD_COVER', 10)) \
    .median() \
    .select('ST_B10') # Thermal band

# Sentinel-2 (Land Use / Land Cover)
sentinel = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(ward_bbox) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .median()

# GHSL (Built-up Density)
ghsl = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S") \
    .filterBounds(ward_bbox) \
    .median()

# 4. Export to Google Drive at 100m Scale
# You can duplicate this task block for Sentinel-2, ERA5, and GHSL
print("Starting GEE Export Task...")
task = ee.batch.Export.image.toDrive(
    image=landsat.clip(ward_bbox),
    description='Ward1_Landsat_LST',
    folder='SIH_UrbanHeat',
    scale=100, # Forces the output into your 100m analytical grid
    region=ward_bbox,
    crs='EPSG:4326'
)
task.start()

# 5. Pull OpenStreetMap Data via OSMnx
print("Fetching OSM Data...")

# Fetch street network for the bounding box
G = ox.graph_from_bbox((west,south,east,north),network_type='all')
roads = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Fetch building polygons
tags = {'building': True}
buildings = ox.features_from_bbox(bbox = (west,south,east,north), tags = tags)

# Save vector data locally to the data folder
roads.to_file('data/ward1_roads.geojson', driver='GeoJSON')
buildings.to_file('data/ward1_buildings.geojson', driver='GeoJSON')
print("OSM Data Extracted and Saved.")