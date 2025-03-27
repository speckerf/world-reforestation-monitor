library(dplyr)
library(ggplot2)
library(cetcolor)
library(cowplot)
library(scales) # For rescale


BIOME_description <- readr::read_csv('../data/misc/ecoregion_biome_table.csv') %>% dplyr::select(c('BIOME_NUM', 'BIOME_NAME')) %>% dplyr::distinct()

# Create a new column with shortened versions of the BIOME_NAME
BIOME_description <- BIOME_description %>%
  mutate(
    BIOME_SHORT = case_when(
      BIOME_NAME == "Tundra" ~ "Tundra",
      BIOME_NAME == "Tropical & Subtropical Moist Broadleaf Forests" ~ "Trop. Moist Forests",
      BIOME_NAME == "Mediterranean Forests, Woodlands & Scrub" ~ "Mediterranean",
      BIOME_NAME == "Deserts & Xeric Shrublands" ~ "Deserts",
      BIOME_NAME == "Temperate Grasslands, Savannas & Shrublands" ~ "Temp. Grasslands",
      BIOME_NAME == "Boreal Forests/Taiga" ~ "Boreal Forests",
      BIOME_NAME == "Temperate Conifer Forests" ~ "Temp. Conifer Forests",
      BIOME_NAME == "Temperate Broadleaf & Mixed Forests" ~ "Temp. Broadl. Forests",
      BIOME_NAME == "Montane Grasslands & Shrublands" ~ "Montane Grasslands",
      BIOME_NAME == "Mangroves" ~ "Mangroves",
      BIOME_NAME == "Flooded Grasslands & Savannas" ~ "Flooded Grasslands",
      BIOME_NAME == "Tropical & Subtropical Grasslands, Savannas & Shrublands" ~ "Trop. Grasslands",
      BIOME_NAME == "Tropical & Subtropical Dry Broadleaf Forests" ~ "Trop. Dry Forests",
      BIOME_NAME == "Tropical & Subtropical Coniferous Forests" ~ "Trop. Conifer Forests",
      TRUE ~ "N/A"
    )
  )

coefficient_of_determination <- function(true, pred) {
  ss_total <- sum((true - mean(true))^2)
  ss_residual <- sum((true - pred)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  return(r_squared)
}

coefficient_of_determination_oos <- function(true, pred, global_true) {
  ss_total <- sum((true - mean(global_true))^2)
  ss_residual <- sum((true - pred)^2)
  r_squared_oos <- 1 - (ss_residual / ss_total)
  return(r_squared_oos)
}



# Define additional metric functions
mean_absolute_error <- function(true, pred) {
  mean(abs(true - pred))
}

# Define additional metric functions
mean_error <- function(true, pred) {
  mean(pred - true)
}

root_mean_squared_error <- function(true, pred) {
  sqrt(mean((true - pred)^2))
}

# norm_root_mean_squared_error <- function(true, pred) {
#   sqrt(mean((true - pred)^2)) / mean(true)
# }

mean_error <- function(true, pred) {
  mean(pred - true)
}

uncertainty_agreement_ratio <- function(true, pred, variable) {
  # Check input validity
  stopifnot(variable %in% c('lai', 'fapar', 'fcover'))
  
  # Calculate absolute error
  abs_error <- abs(true - pred)
  
  # Define threshold based on variable
  threshold <- switch(variable,
                      'fapar'  = pmax(0.10 * true, 0.05),
                      'fcover' = pmax(0.10 * true, 0.05),
                      'lai'    = pmax(0.20 * true, 0.5)
  )
  
  # Determine which predictions are within uncertainty bounds
  within_bounds <- abs_error <= threshold
  
  # Calculate agreement ratio
  ratio <- sum(within_bounds, na.rm = TRUE) / length(true)
  
  return(ratio)
}


# Function to read and prepare data
load_data <- function(file_path, biome_desc, top_n = 6) {
  data <- readr::read_csv(file_path) %>%
    left_join(biome_desc)
  return(data)
}

# select top biomes
subset_biomes <- function(data, top_n = 6){
  top_biomes <- data %>%
    count(BIOME_SHORT) %>%
    arrange(desc(n)) %>%
    slice_head(n = top_n)
  
  data_subset <- data %>%
    filter(BIOME_SHORT %in% top_biomes$BIOME_SHORT) %>%
    mutate(BIOME_SHORT = factor(BIOME_SHORT, levels = top_biomes$BIOME_SHORT))
  
  data_subset
}


# Function to calculate metrics
calculate_metrics <- function(data, variable) {
  list(
    N = nrow(data),
    R2 = coefficient_of_determination(data$true, data$pred),
    RMSE = root_mean_squared_error(data$true, data$pred),
    nRMSE = root_mean_squared_error(data$true, data$pred) / abs(max(data$true) - min(data$true)),
    MAE = mean_absolute_error(data$true, data$pred),
    ME = mean_error(data$true, data$pred),
    UAR = uncertainty_agreement_ratio(data$true, data$pred, variable)
  )
}


add_log_scale_colors <- function(data, x_col, y_col, bandwidth = NULL, color_palette = cetcolor::cet_pal(256, name = "l17")[20:256]) {
  library(dplyr)
  library(MASS) # For kde2d
  library(scales) # For rescale
  
  # Check if the specified columns exist in the dataframe
  if (!(x_col %in% colnames(data)) || !(y_col %in% colnames(data))) {
    stop("Specified columns do not exist in the dataframe.")
  }
  
  # Compute 2D density
  density_info <- kde2d(data[[x_col]], data[[y_col]], n = 100, h = bandwidth)
  
  # Approximate density values for each point
  data$density <- with(data, 
                       approx(density_info$x, seq_along(density_info$x), xout = data[[x_col]])$y) *
    approx(density_info$y, seq_along(density_info$y), xout = data[[y_col]])$x
  
  # Apply log transformation
  data$log_density <- log1p(data$density)
  
  # Normalize log densities to [0, 1]
  data$normalized_log_density <- rescale(data$log_density)
  
  # Map normalized densities to the color palette
  color_function <- colorRampPalette(color_palette)
  data$colors <- color_function(100)[cut(data$normalized_log_density, breaks = 100)]
  
  return(data)
}
# 
# data
# density_info <- kde2d(data[['true']], data[['pred']], n = 100, h = 0.01)
# data$density <- with(data, 
#                      approx(density_info$x, seq_along(density_info$x), xout = data[["true"]])$y) *
#   approx(density_info$y, seq_along(density_info$y), xout = data[["pred"]])$x
# 
# data$log_density <- log1p(data$density)
# 
# data$normalized_log_density <- rescale(data$log_density)
# color_function <- colorRampPalette(cetcolor::cet_pal(256, name = "l17")[20:256])
# data$colors <- color_function(100)[cut(data$normalized_log_density, breaks = 100)]
# 
# data %>%
#   ggplot(aes(x = true, y = pred)) +
#   geom_point(color = data$colors, size = 0.9)

# Function to create global scatterplot
create_global_plot <- function(data, metrics, xlim, ylim, density_palette, density_bandwidth, axis_prefix, variable) {
  data <- data %>%
        mutate(colors = densCols(true, pred, colramp = colorRampPalette(density_palette), bandwidth = density_bandwidth))
  
  # position_x = 90% of x-axis
  # position_y = 10 % of y-axis
  position_x = xlim[1] + 0.95 * (xlim[2] - xlim[1])
  position_y = ylim[1] + 0.05 * (ylim[2] - ylim[1])
  
  # Add uncertainty bound lines depending on variable
  uncertainty_fun <- switch(variable,
                            "fapar"  = function(x) pmax(0.10 * x, 0.05),
                            "fcover" = function(x) pmax(0.10 * x, 0.05),
                            "lai"    = function(x) pmax(0.20 * x, 0.5),
                            stop("Invalid variable")
  )
  
  x_vals <- seq(xlim[1], xlim[2], length.out = 200)
  offset <- uncertainty_fun(x_vals)
  
  df_bounds <- data.frame(
    x = x_vals,
    upper = x_vals + offset,
    lower = x_vals - offset
  )
  
  
  p <- data %>%
    ggplot(aes(x = true, y = pred)) +
    geom_point(color = data$colors, size = 0.9) +
    # geom_point(aes(color = data$BIOME_PLOT), size = 0.9) +
    geom_line(data = df_bounds, aes(x = x, y = x), linetype = "dashed", color = "gray40", linewidth = 1.0) +
    geom_line(data = df_bounds, aes(x = x, y = upper), linetype = "dashed", color = "gray40", linewidth = 0.5) +
    geom_line(data = df_bounds, aes(x = x, y = lower), linetype = "dashed", color = "gray40", linewidth = 0.5) + 
    # geom_abline(intercept = 0, slope = 1, color = "#4D4D4D", linetype = "dashed", linewidth = 1.0) +
    # geom_smooth(method = "loess", se = TRUE, color = "#4D4D4D", linewidth = 0.75, method.args = list(family = "symmetric")) + 
    labs(x = paste0(axis_prefix, "In-situ measurement"), y = paste0(axis_prefix, "S2 retrieval")) +
    coord_fixed() +
    scale_x_continuous(limits = xlim) +
    scale_y_continuous(limits = ylim) +
    theme_minimal() +
    annotate(
      "label", x = position_x, y = position_y,
      label = paste("N =", metrics$N,
                    "\nR² =", round(metrics$R2, 3),
                    "\nRMSE =", round(metrics$RMSE, 3),
                    "\nMAE =", round(metrics$MAE, 3),
                    "\nUAR =", round(metrics$UAR * 100, 1), "%"),
      hjust = 0.75, vjust = 0.25, size = 3.5,
      fill = "white", alpha = 0.7, label.size = NA
    )
  p
  # p + 
  #   geom_line(data = df_bounds, aes(x = x, y = upper), linetype = "dashed", color = "gray40") +
  #   geom_line(data = df_bounds, aes(x = x, y = lower), linetype = "dashed", color = "gray40")
  
}

# Function to create biome-specific plot
create_biome_plot <- function(data, palette, xlim, ylim, density_palette, density_bandwidth, axis_prefix, variable) {
  
  # Add uncertainty bound lines depending on variable
  uncertainty_fun <- switch(variable,
                            "fapar"  = function(x) pmax(0.10 * x, 0.05),
                            "fcover" = function(x) pmax(0.10 * x, 0.05),
                            "lai"    = function(x) pmax(0.20 * x, 0.5),
                            stop("Invalid variable")
  )
  
  x_vals <- seq(xlim[1], xlim[2], length.out = 200)
  offset <- uncertainty_fun(x_vals)
  
  df_bounds <- data.frame(
    x = x_vals,
    upper = x_vals + offset,
    lower = x_vals - offset
  )
  
  
  data <- data %>%
    group_by(BIOME_SHORT) %>%
    mutate(
      colors_by_biome = densCols(
        true, pred,
        colramp = colorRampPalette(density_palette),
        bandwidth = density_bandwidth
      )
    ) %>%
    ungroup()
  
  fit_values_biome <- data %>%
    group_by(BIOME_SHORT) %>%
    summarize(
      R2 = coefficient_of_determination_oos(true, pred, data[['true']]),
      MAE = mean_absolute_error(true, pred),
      ME = mean_error(true, pred),
      RMSE = root_mean_squared_error(true, pred),
      UAR = uncertainty_agreement_ratio(true, pred, variable),
      # MAPE = mean(abs((true- pred) / true) * 100),
      nRMSE = RMSE / abs(max(true, na.rm = TRUE) - min(true, na.rm = TRUE)),
      N = n()
    )
  
  print(fit_values_biome)
  
  # position_x = 90% of x-axis
  # position_y = 10 % of y-axis
  position_x = xlim[1] + 0.9 * (xlim[2] - xlim[1])
  position_y = ylim[1] + 0.1 * (ylim[2] - ylim[1])
  
  
  data %>%
    ggplot(aes(x = true, y = pred)) +
    geom_point(aes(color = colors_by_biome), size = 0.6) +
    scale_color_identity() +
    # geom_abline(intercept = 0, slope = 1, color = "#4D4D4D", linetype = "dashed", linewidth = 1.0) +
    geom_line(data = df_bounds, aes(x = x, y = x), linetype = "dashed", color = "gray40", linewidth = 0.8) +
    geom_line(data = df_bounds, aes(x = x, y = upper), linetype = "dashed", color = "gray40",  linewidth = 0.4) +
    geom_line(data = df_bounds, aes(x = x, y = lower), linetype = "dashed", color = "gray40", linewidth = 0.4) + 
    facet_wrap(~ BIOME_SHORT, ncol = 3) +
    # geom_smooth(method = "lm", color = "#4D4D4D", linewidth = 0.75) +
    # geom_smooth(method = "loess", se = TRUE, color = "#4D4D4D", linewidth = 0.75, method.args = list(family = "symmetric")) + 
    # geom_ribbon(aes(ymin = lower, ymax = upper, x = true), alpha = 0.2, fill = "#4D4D4D") +
    scale_x_continuous(limits = xlim) +
    scale_y_continuous(limits = ylim) +
    coord_fixed() +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines")) + 
    labs(x = paste0(axis_prefix, "In-situ measurement"), y = paste0(axis_prefix, "S2 retrieval")) +
    geom_label(
      data = fit_values_biome,
      aes(
        x = position_x, y = position_y,
        label = paste("RMSE =", round(RMSE, 3),
                      "\nMAE =", round(MAE, 3),
                      "\nUAR =", round(UAR * 100, 1), "%")
      ),
      hjust = 0.75, vjust = 0.25, size = 2.75, inherit.aes = FALSE,
      fill = "white", alpha = 0.7, label.size = NA
    )
}

# Main workflow
file_paths <- list(
  lai = "data/ensemble_predictions_lai.csv",
  fapar = "data/ensemble_predictions_fapar.csv",
  fcover = "data/ensemble_predictions_fcover.csv"
)

palette <- cetcolor::cet_pal(256, name = "l4")[1:256]
palette <- cetcolor::cet_pal(256, name = "l17")[20:256]

# Function to create a log-transformed color palette
create_sqrt_palette <- function(original_palette, n = 256) {
  # Total number of colors in the original palette
  original_length <- length(original_palette)
  
  # Generate log-spaced indices
  sqrt_indices <- sqrt(seq(0, original_length - 1)) # log1p ensures no log(0) issues
  sqrt_indices <- rescale(sqrt_indices, to = c(1, original_length)) # Rescale to palette indices
  
  # Interpolate the colors using the log indices
  color_function <- colorRampPalette(original_palette)
  log_palette <- color_function(n)[as.integer(sqrt_indices)]
  
  return(log_palette)
}

sqrt_palette <- create_sqrt_palette(palette)

####
# Plot LAI
####
data <- load_data(file_paths[["lai"]], BIOME_description)
metrics <- calculate_metrics(data, 'lai')
# data <- assign_density_colors(data, palette)

p1 <- create_global_plot(data, metrics, xlim = c(0, 6), ylim = c(0, 5), density_palette = palette, density_bandwidth = 0.2, axis_prefix = 'LAIe - ', variable = 'lai')

data_subset <- subset_biomes(data, top_n = 6)
# metrics <- 
p2 <- create_biome_plot(data = data_subset, xlim = c(0, 6), ylim = c(0, 5), density_palette = palette, density_bandwidth = 0.2, axis_prefix = 'LAIe - ', variable = 'lai')

plot_grid(p1, p2, rel_widths = c(3, 4))

####
# Plot FAPAR
####

data <- load_data(file_paths[["fapar"]], BIOME_description)
metrics <- calculate_metrics(data, 'fapar')
p3 <- create_global_plot(data, metrics, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, axis_prefix = 'FAPAR - ', variable = 'fapar')

data_subset <- subset_biomes(data, top_n = 6)
# metrics <- 
p4 <- create_biome_plot(data = data_subset, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, axis_prefix = 'FAPAR - ', variable = 'fapar')
plot_grid(p3, p4, rel_widths = c(3, 4))



# plot_grid(p1, p2, p3, p4, rel_widths = c(3, 4))



####
# Plot FCOVER
####

data <- load_data(file_paths[["fcover"]], BIOME_description)
metrics <- calculate_metrics(data, 'fcover')
p5 <- create_global_plot(data, metrics, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, axis_prefix = 'FCOVER - ', variable = 'fcover')

data_subset <- subset_biomes(data, top_n = 6)
# metrics <- 
p6 <- create_biome_plot(data = data_subset, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, axis_prefix = 'FCOVER - ', variable = 'fcover')
plot_grid(p5, p6, rel_widths = c(3, 4))


####
# Plot All
####

plot_grid(p1, p2, p3, p4, p5, p6, rel_widths = c(3, 4), ncol = 2, labels = c("A1", "B1", "A2", "B2", "A3", "B3"))
# export 750 x 1000

