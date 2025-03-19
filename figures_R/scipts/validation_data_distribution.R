# Install and load necessary packages
# install.packages(c("ggplot2", "sf", "rnaturalearth", "rnaturalearthdata"))
library(ggplot2)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(ggrepel)
library(dplyr)

# Load the CSV file
file_path <- "../data/validation_pipeline/output/lai/EXPORT_GBOV_RM6,7_20240620120826_all_reflectances_with_angles.csv"
df <- read.csv(file_path)

# Extract coordinates from the '.geo' column
df$coordinates <- lapply(df$.geo, function(x) {
  jsonlite::fromJSON(x)$coordinates
})

# Split coordinates into longitude and latitude
df$longitude <- sapply(df$coordinates, `[`, 1)
df$latitude <- sapply(df$coordinates, `[`, 2)

# Remove duplicate SITE_IDs
df_unique <- df %>% dplyr::group_by(SITE_ID) %>% dplyr::slice(1)

# Convert to an sf object
df_sf <- st_as_sf(df_unique, coords = c("longitude", "latitude"), crs = 4326)


# Get the world map data
world <- ne_countries(scale = "medium", returnclass = "sf")


# Create the plot
p <- ggplot(data = world) +
  geom_sf(fill = "lightgray") +
  geom_sf(data = df_sf, size = 1) +  # Adjust size of the dots
  geom_text_repel(data = df_unique, aes(x = longitude, y = latitude, label = SITE_ID),
                  size = 3.0, fontface = "bold", max.overlaps = Inf,  # Smaller and bolder text annotations
                  segment.size = 0.5, segment.color = "grey50") + # Add repelling labels with smaller grey lines
  theme(axis.title.x = element_blank(), axis.title.y = element_blank())  # Hide x and y axis labels

# save plot
# Save the plot as a high-resolution PNG file
ggsave("output/validation_data_distribution.png", plot = p, width = 10, height = 7, dpi = 300)



# number of ecoregions:
eco_ids <- df['ECO_ID'] %>% dplyr::distinct()

# load the ecoregion_biome table to find out how many of the 14 biomes are represented:
biome_table_path = "../data/misc/ecoregion_biome_table.csv"
biome_table <- readr::read_csv(biome_table_path)

val_biome_table <- eco_ids %>% dplyr::left_join(biome_table %>% dplyr::select("ECO_ID", "BIOME_NUM", "BIOME_NAME"))
val_biome_table %>% dplyr::select('BIOME_NUM') %>% dplyr::distinct()
