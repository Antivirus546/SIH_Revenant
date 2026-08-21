import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from rasterstats import zonal_stats

# 1. Define Bounding Box & Grid Size (~100m)
west, south, east, north = 77.58, 12.97, 77.62, 13.00
grid_size = 0.0009  

# 2. Create the 100m Fishnet Grid
print("Generating 100m grid...")
grid_cells = []
for x0 in np.arange(west, east, grid_size):
    for y0 in np.arange(south, north, grid_size):
        grid_cells.append(box(x0, y0, x0 + grid_size, y0 + grid_size))

grid = gpd.GeoDataFrame({'grid_id': range(len(grid_cells))}, geometry=grid_cells, crs="EPSG:4326")

# 3. Load & Filter Vector Data
print("Loading and filtering map vectors...")
roads = gpd.read_file('data/ward1_roads.geojson')
buildings = gpd.read_file('data/ward1_buildings.geojson')

# Filter out points/invalid geometries to prevent intersection crashes
buildings = buildings[buildings.geom_type.isin(['Polygon', 'MultiPolygon'])]
roads = roads[roads.geom_type.isin(['LineString', 'MultiLineString'])]

# 4. Extract Building Area per Grid Cell
print("Calculating building densities...")
bldg_inter = gpd.overlay(buildings, grid, how='intersection')
bldg_inter['bldg_area_sqm'] = bldg_inter.to_crs('EPSG:32643').geometry.area 
bldg_stats = bldg_inter.groupby('grid_id')['bldg_area_sqm'].sum().reset_index()

# 5. Extract Road Length per Grid Cell
print("Calculating road networks...")
road_inter = gpd.overlay(roads, grid, how='intersection')
road_inter['road_length_m'] = road_inter.to_crs('EPSG:32643').geometry.length
road_stats = road_inter.groupby('grid_id')['road_length_m'].sum().reset_index()

# 6. Extract Temperature & Indices from Landsat Image
print("Extracting raster pixels (Temp, NDVI, NDBI, Albedo)...")
tif_path = 'data/Ward1_Landsat_Indices.tif'

# Band 1: Target_Temp
temp_stats = zonal_stats(grid, tif_path, band=1, stats='median')
grid['target_temp'] = [stat['median'] for stat in temp_stats]

# Band 2: NDVI
ndvi_stats = zonal_stats(grid, tif_path, band=2, stats='median')
grid['ndvi'] = [stat['median'] for stat in ndvi_stats]

# Band 3: NDBI
ndbi_stats = zonal_stats(grid, tif_path, band=3, stats='median')
grid['ndbi'] = [stat['median'] for stat in ndbi_stats]

# Band 4: Albedo
albedo_stats = zonal_stats(grid, tif_path, band=4, stats='median')
grid['albedo'] = [stat['median'] for stat in albedo_stats]

# 7. Merge into Final Tabular Format
print("Merging dataset...")
final_df = grid.merge(bldg_stats, on='grid_id', how='left')
final_df = final_df.merge(road_stats, on='grid_id', how='left')

# Replace missing values with 0
final_df = final_df.fillna(0)

# Save to CSV
final_df.drop(columns='geometry').to_csv('data/ward1_processed.csv', index=False)
print("Success! Updated tabular data saved to data/ward1_processed.csv")