import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from rasterstats import zonal_stats

# 1. Define Bounding Box & Grid Size (0.0009 degrees is ~100 meters)
north, south, east, west = 13.00, 12.97, 77.62, 77.58
grid_size = 0.0009  

# 2. Create the 100m Fishnet Grid
print("Generating 100m grid...")
grid_cells = []
for x0 in np.arange(west, east, grid_size):
    for y0 in np.arange(south, north, grid_size):
        grid_cells.append(box(x0, y0, x0 + grid_size, y0 + grid_size))

grid = gpd.GeoDataFrame({'grid_id': range(len(grid_cells))}, geometry=grid_cells, crs="EPSG:4326")

# 3. Load Vector Data
print("Loading map vectors...")
roads = gpd.read_file('data/ward1_roads.geojson')
buildings = gpd.read_file('data/ward1_buildings.geojson')

buildings = buildings[buildings.geom_type.isin(['Polygon', 'MultiPolygon'])]
roads = roads[roads.geom_type.isin(['LineString', 'MultiLineString'])]

# 4. Extract Building Area per Grid Cell
print("Calculating building densities...")
bldg_inter = gpd.overlay(buildings, grid, how='intersection')
# Convert to UTM projection (EPSG:32643) to calculate area in accurate square meters
bldg_inter['bldg_area_sqm'] = bldg_inter.to_crs('EPSG:32643').geometry.area 
bldg_stats = bldg_inter.groupby('grid_id')['bldg_area_sqm'].sum().reset_index()

# 5. Extract Road Length per Grid Cell
print("Calculating road networks...")
road_inter = gpd.overlay(roads, grid, how='intersection')
road_inter['road_length_m'] = road_inter.to_crs('EPSG:32643').geometry.length
road_stats = road_inter.groupby('grid_id')['road_length_m'].sum().reset_index()

# 6. Extract Temperature from Landsat Image (Zonal Statistics)
print("Extracting temperature pixels...")
# Ensure 'Ward1_Landsat_LST.tif' is downloaded from Drive and inside your data folder
temp_stats = zonal_stats(grid, 'data/Ward1_Landsat_LST.tif', stats='median')
grid['target_temp'] = [stat['median'] for stat in temp_stats]

# 7. Merge into Final Tabular Format
print("Merging dataset...")
final_df = grid.merge(bldg_stats, on='grid_id', how='left')
final_df = final_df.merge(road_stats, on='grid_id', how='left')

# Replace missing values (empty fields) with 0
final_df = final_df.fillna(0)

# Save to CSV (dropping the shape column since XGBoost only reads numbers)
final_df.drop(columns='geometry').to_csv('data/ward1_processed.csv', index=False)
print("Success! Tabular data saved to data/ward1_processed.csv")