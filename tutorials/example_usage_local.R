# Load the required libraries
library(terra)
library(stringr)

# 1: Setup
data_dir <- "/Volumes/RAID/felix_oemc/results_100m"

variables <- c("fapar", "lai", "fcover")
bands <- c("mean", "std", "count")
years <- 2019:2024

pattern <- "^(fcover|fapar|lai)_rtm\\.mlp_(mean|std|count)_100m_s_(\\d{8})_(\\d{8})_go_epsg\\.4326_v\\d{2}\\.tif$"

# 2: Load data lazily
# List all files in the directory
files <- list.files(data_dir, pattern = "\\.tif$", full.names = TRUE)

# Initialize an empty list
file_names <- list()
datasets <- list()

# Loop through variables and bands correctly
for (var in variables) {
  for (band in bands) {
    key <- paste0(var, "_", band)  # Create a key for the list
    file_names[[key]] <- files[str_detect(files, paste0(var, "_rtm\\.mlp_", band))]  # Correct pattern match
    datasets[[key]] <- rast(file_names[[key]])  # Load the raster
  }
}


# 3: Visualize
variable <- "lai_mean"
bbox <- c(xmin = 8, xmax = 9, ymin = 47, ymax = 48)  # Example bounding box

# Extract the LAI mean raster stack
stack <- datasets[[variable]]

# Crop to bounding box
crop <- crop(stack, ext(bbox))

# Set up multi-panel plotting
plot(crop, col = c("#fffdcd","#e1cd73","#aaac20","#5f920c","#187328","#144b2a","#172313"))


