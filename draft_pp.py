# download the info about Porto from openstreetmap, and find the boundaries of city


import osmnx as ox
import matplotlib.pyplot as plt
import contextily as cx

# Download the Porto boundary
porto = ox.geocode_to_gdf("Porto, Portugal")

# Reproject to Web Mercator (required for map tiles)
porto_web_mercator = porto.to_crs(epsg=3857)

# Plot with basemap
fig, ax = plt.subplots(figsize=(10, 10))
porto_web_mercator.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)
cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)  # OSM map background
ax.set_title("Porto Boundary on OpenStreetMap")
plt.axis("off")
plt.show()


# Download the administrative boundary of Porto, Portugal
porto_boundary = ox.geocode_to_gdf("Porto, Portugal")

boundary = {
    'lati_min': porto_boundary['bbox_south'].iloc[0],
    'lati_max': porto_boundary['bbox_north'].iloc[0],
    'long_min': porto_boundary['bbox_west'].iloc[0],
    'long_max': porto_boundary['bbox_east'].iloc[0]
}