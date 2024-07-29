# command line script:
# invoke with:
# Rscript soil_insitu_database_convert_to_parquet.R /path/to/data/directory
# path to R installation: "/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources"

# Load required packages
library(qs)
library(arrow)
library(tidyverse)

# Function to convert .qs files to .parquet
convert_qs_to_parquet <- function(data_dir) {
  # List all .qs files in the directory
  qs_files <- list.files(path = data_dir, pattern = "\\.qs$", full.names = TRUE)
  
  # Iterate over each .qs file and convert to .parquet
  for (qs_file in qs_files) {
    # Extract the base name without extension
    base_name <- tools::file_path_sans_ext(basename(qs_file))
    
    # Define the output .parquet file path
    parquet_file <- file.path(data_dir, paste0(base_name, ".parquet"))
    
    # Read the .qs file and write to .parquet
    data <- qread(qs_file)
    write_parquet(data, parquet_file)
    
    cat("Converted:", qs_file, "to", parquet_file, "\n")
  }
}

# Main execution
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("No data directory provided. Please provide the path to the data directory as an argument.")
} else {
  data_dir <- args[1]
  convert_qs_to_parquet(data_dir)
}
