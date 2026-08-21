import ee
import osmnx as ox
import geopandas as gpd

# 1. Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project='spring-radar-478010-k1')

# 2. Define Bengaluru Ward 1 Bounding Box 
west, south, east, north = 77.58, 12.97, 77.62, 13.00
ward_bbox = ee.Geometry.BBox(west, south, east, north)

# Set timeframe to peak summer months to capture high heat stress
start_date = '2025-03-01'
end_date = '2025-05-31'

# 3. Pull GEE Datasets
print("Querying GEE Datasets...")

# Landsat 8 (Surface Temperature, NDVI, NDBI, Albedo)
# We need bands 4 (Red), 5 (NIR), 6 (SWIR1), and 10 (Thermal)
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(ward_bbox) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUD_COVER', 10)) \
    .median()

# Calculate NDVI (Normalized Difference Vegetation Index)
# NDVI = (NIR - Red) / (NIR + Red)
ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

# Calculate NDBI (Normalized Difference Built-up Index)
# NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
ndbi = landsat.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')

# Calculate a simple Albedo approximation for Landsat 8
# Albedo ~ (0.356 * Blue) + (0.130 * Red) + (0.373 * NIR) + (0.085 * SWIR1) + (0.072 * SWIR2) - 0.0018
# We'll use a simplified version utilizing the bands we have easily accessible in this collection for a quick index.
albedo = landsat.expression(
    '(0.356 * B2) + (0.130 * B4) + (0.373 * B5) + (0.085 * B6) + (0.072 * B7) - 0.0018',
    {
        'B2': landsat.select('SR_B2').multiply(0.0000275).add(-0.2),
        'B4': landsat.select('SR_B4').multiply(0.0000275).add(-0.2),
        'B5': landsat.select('SR_B5').multiply(0.0000275).add(-0.2),
        'B6': landsat.select('SR_B6').multiply(0.0000275).add(-0.2),
        'B7': landsat.select('SR_B7').multiply(0.0000275).add(-0.2)
    }
).rename('Albedo')

# Combine all indices and temperature into a single image
final_image = landsat.select('ST_B10').rename('Target_Temp').addBands([ndvi, ndbi, albedo]).toFloat()

# 4. Export to Google Drive at 100m Scale
print("Starting GEE Export Task...")
task = ee.batch.Export.image.toDrive(
    image=final_image.clip(ward_bbox),
    description='Ward1_Landsat_Indices',
    folder='SIH_UrbanHeat',
    scale=100, 
    region=ward_bbox,
    crs='EPSG:4326'
)
task.start()

# 5. Pull OpenStreetMap Data via OSMnx
print("Fetching OSM Data...")

# Fetch street network for the bounding box
G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='all')
roads = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Fetch building polygons
tags = {'building': True}
buildings = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)

# Save vector data locally to the data folder
roads.to_file('data/ward1_roads.geojson', driver='GeoJSON')
buildings.to_file('data/ward1_buildings.geojson', driver='GeoJSON')
print("OSM Data Extracted and Saved.")