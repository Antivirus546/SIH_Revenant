import ee
import osmnx as ox
import geopandas as gpd

# 1. Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project='spring-radar-478010-k1')

# 2. Define Bengaluru Ward 1 Bounding Box 
west, south, east, north = 77.58, 12.97, 77.62, 13.00
ward_bbox = ee.Geometry.BBox(west, south, east, north)

start_date = '2025-03-01'
end_date = '2025-05-31'

# 3. Pull GEE Datasets
print("Querying GEE Datasets...")

# Landsat 8 
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(ward_bbox) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUD_COVER', 10)) \
    .median()

# Calculate Indices
ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
ndbi = landsat.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
ndwi = landsat.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')

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

# Fetch SRTM Elevation Data
elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('Elevation')

# Combine all 6 bands and force 32-bit float format to prevent export errors
final_image = landsat.select('ST_B10').rename('Target_Temp').addBands([ndvi, ndbi, albedo, ndwi, elevation]).toFloat()

# 4. Export to Google Drive
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

# 5. Pull OpenStreetMap Data
print("Fetching OSM Data...")
G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='all')
roads = ox.graph_to_gdfs(G, nodes=False, edges=True)

tags = {'building': True}
buildings = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)

roads.to_file('data/ward1_roads.geojson', driver='GeoJSON')
buildings.to_file('data/ward1_buildings.geojson', driver='GeoJSON')
print("OSM Data Extracted and Saved.")