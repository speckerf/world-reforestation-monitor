library(dplyr)
library(ggplot2)
library(cetcolor)
library(cowplot)
library(scales) # For rescale


coefficient_of_determination <- function(true, pred) {
  ss_total <- sum((true - mean(true))^2)
  ss_residual <- sum((true - pred)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  return(r_squared)
}

coefficient_of_determination_global <- function(true, pred, global_true) {
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

mean_error <- function(true, pred) {
  mean(pred - true)
}

uncertainty_agreement_ratio <- function(true, pred, variable) {
  # Check input validity
  stopifnot(variable %in% c('laie', 'fapar', 'fcover'))
  
  # Calculate absolute error
  abs_error <- abs(true - pred)
  
  # Define threshold based on variable
  threshold <- switch(variable,
                      'fapar'  = pmax(0.2 * true, 0.1),
                      'fcover' = pmax(0.2 * true, 0.1),
                      'laie'    = pmax(0.2 * true, 1.0)
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

# select top NLCDs
subset_lcs <- function(data, top_n = 6){
  assertthat::assert_that('NLCD' %in% colnames(data))
  top_lc <- data %>%
    count(NLCD) %>%
    arrange(desc(n)) %>%
    slice_head(n = top_n)
  
  data_subset <- data %>%
    filter(NLCD %in% top_lc$NLCD) %>%
    mutate(NLCD = factor(NLCD, levels = top_lc$NLCD))
  
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




# Function to create global scatterplot
create_global_plot <- function(data, metrics, xlim, ylim, density_palette, density_bandwidth, variable, y_label, x_label, breaks = NULL) {
  data <- data %>%
    mutate(colors = densCols(true, pred, colramp = colorRampPalette(density_palette), bandwidth = density_bandwidth))
  
  # position_x = 90% of x-axis
  # position_y = 10 % of y-axis
  position_x = xlim[1] + 0.95 * (xlim[2] - xlim[1])
  position_y = ylim[1] + 0.05 * (ylim[2] - ylim[1])
  
  # Add uncertainty bound lines depending on variable
  uncertainty_fun <- switch(variable,
                            "fapar"  = function(x) pmax(0.2 * x, 0.1),
                            "fcover" = function(x) pmax(0.2 * x, 0.1),
                            "laie"    = function(x) pmax(0.2 * x, 1.0),
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
    geom_line(data = df_bounds, aes(x = x, y = x), linetype = "dashed", color = "black", linewidth = 1.0) +
    geom_line(data = df_bounds, aes(x = x, y = upper), linetype = "dashed", color = "black", linewidth = 0.5) +
    geom_line(data = df_bounds, aes(x = x, y = lower), linetype = "dashed", color = "black", linewidth = 0.5) + 
    # geom_abline(intercept = 0, slope = 1, color = "#4D4D4D", linetype = "dashed", linewidth = 1.0) +
    # geom_smooth(method = "loess", se = TRUE, color = "#4D4D4D", linewidth = 0.75, method.args = list(family = "symmetric")) + 
    labs(x = x_label, y = y_label) +
    coord_fixed() +
    scale_x_continuous(limits = xlim, breaks = breaks) +
    scale_y_continuous(limits = ylim, breaks = breaks) +
    theme_minimal() +
    annotate(
      "label", x = position_x, y = position_y,
      label = paste("N =", metrics$N,
                    "\nR² =", round(metrics$R2, 3),
                    "\nRMSE =", round(metrics$RMSE, 3),
                    "\nMAE =", round(metrics$MAE, 3),
                    "\nUAR =", round(metrics$UAR * 100, 1), "%"),
      hjust = 0.75, vjust = 0.25, size = 2.5,
      fill = "white", alpha = 0.7, label.size = NA
    )
  p
  # p + 
  #   geom_line(data = df_bounds, aes(x = x, y = upper), linetype = "dashed", color = "gray40") +
  #   geom_line(data = df_bounds, aes(x = x, y = lower), linetype = "dashed", color = "gray40")
  
}

# Function to create biome-specific plot
create_lc_plot <- function(data, xlim, ylim, density_palette, density_bandwidth, x_label, y_label, variable, breaks = NULL, ncol_facet = 3) {
  
  # Add uncertainty bound lines depending on variable
  uncertainty_fun <- switch(variable,
                            "fapar"  = function(x) pmax(0.2 * x, 0.1),
                            "fcover" = function(x) pmax(0.2 * x, 0.1),
                            "laie"    = function(x) pmax(0.20 * x, 1.0),
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
    group_by(NLCD) %>%
    mutate(
      colors_by_lc = densCols(
        true, pred,
        colramp = colorRampPalette(density_palette),
        bandwidth = density_bandwidth
      )
    ) %>%
    ungroup()
  
  fit_values_lc <- data %>%
    group_by(NLCD) %>%
    summarize(
      R2 = coefficient_of_determination_global(true, pred, data[['true']]),
      MAE = mean_absolute_error(true, pred),
      ME = mean_error(true, pred),
      RMSE = root_mean_squared_error(true, pred),
      UAR = uncertainty_agreement_ratio(true, pred, variable),
      # MAPE = mean(abs((true- pred) / true) * 100),
      nRMSE = RMSE / abs(max(true, na.rm = TRUE) - min(true, na.rm = TRUE)),
      N = n()
    )
  
  print(fit_values_lc)
  
  # position_x = 90% of x-axis
  # position_y = 10 % of y-axis
  position_x = xlim[1] + 0.9 * (xlim[2] - xlim[1])
  position_y = ylim[1] + 0.1 * (ylim[2] - ylim[1])
  
  
  data %>%
    ggplot(aes(x = true, y = pred)) +
    geom_point(aes(color = colors_by_lc), size = 0.6) +
    scale_color_identity() +
    # geom_abline(intercept = 0, slope = 1, color = "#4D4D4D", linetype = "dashed", linewidth = 1.0) +
    geom_line(data = df_bounds, aes(x = x, y = x), linetype = "dashed", color = "black", linewidth = 0.8) +
    geom_line(data = df_bounds, aes(x = x, y = upper), linetype = "dashed", color = "black",  linewidth = 0.4) +
    geom_line(data = df_bounds, aes(x = x, y = lower), linetype = "dashed", color = "black", linewidth = 0.4) + 
    facet_wrap(~ NLCD, ncol = ncol_facet) +
    # geom_smooth(method = "lm", color = "#4D4D4D", linewidth = 0.75) +
    # geom_smooth(method = "loess", se = TRUE, color = "#4D4D4D", linewidth = 0.75, method.args = list(family = "symmetric")) + 
    # geom_ribbon(aes(ymin = lower, ymax = upper, x = true), alpha = 0.2, fill = "#4D4D4D") +
    scale_x_continuous(limits = xlim, breaks = breaks) +
    scale_y_continuous(limits = ylim, breaks = breaks) +
    coord_fixed() +
    theme_minimal() +
    theme(panel.spacing = unit(1, "lines")) + 
    labs(x = x_label, y = y_label) +
    geom_label(
      data = fit_values_lc,
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


# palette <- cetcolor::cet_pal(256, name = "l4")[1:256]
palette <- cetcolor::cet_pal(256, name = "l17")[20:256]

file_path <- file.path('..', '..', 'data', 'train_pipeline', 'output', 'predictions_specker_sl2p.csv')
df <- readr::read_csv(file_path)


#### Plot LAI
df_lai <- df %>% dplyr::select(dplyr::matches('laie'), 'NLCD', -dplyr::matches('std')) %>% 
  dplyr::rename(true = laie, pred_sl2p = sl2p_laie_mean, pred_specker = specker_laie_mean)

metrics_sl2p <- calculate_metrics(df_lai %>% dplyr::select(true, pred_sl2p) %>% dplyr::rename(pred = pred_sl2p), 'laie')
metrics_specker <- calculate_metrics(df_lai %>% dplyr::select(true, pred_specker) %>% dplyr::rename(pred = pred_specker), 'laie')

p1 <- create_global_plot(df_lai %>% dplyr::select(true, pred_sl2p) %>% dplyr::rename(pred = pred_sl2p), metrics_sl2p, xlim = c(0, 5), ylim = c(0, 5), density_palette = palette, density_bandwidth = 0.25, y_label = "LAIe - SL2P" , x_label = "LAIe - RMs", variable = 'laie', breaks = c(0, 1,2,3,4,5))
p2 <- create_global_plot(df_lai %>% dplyr::select(true, pred_specker) %>% dplyr::rename(pred = pred_specker), metrics_specker, xlim = c(0, 5), ylim = c(0, 5), density_palette = palette, density_bandwidth = 0.25, y_label = "LAIe - S2BIOPHYS" , x_label = "LAIe - RMs", variable = 'laie', breaks = c(0, 1,2,3,4,5))

df_lai_subset <- df_lai %>% subset_lcs(top_n = 8)
p1_lc <- create_lc_plot(dplyr::select(df_lai_subset, true, pred_sl2p, NLCD) %>% dplyr::rename(pred = pred_sl2p), xlim = c(0, 5), ylim = c(0, 5), density_palette = palette, density_bandwidth = 0.25, x_label = "LAIe - RMs", y_label = "LAIe - SL2P", variable = "laie", breaks = c(0, 1,2,3,4,5), ncol_facet = 4)
p2_lc <- create_lc_plot(dplyr::select(df_lai_subset, true, pred_specker, NLCD) %>% dplyr::rename(pred = pred_specker), xlim = c(0, 5), ylim = c(0, 5), density_palette = palette, density_bandwidth = 0.25, x_label = "LAIe - RMs", y_label = "LAIe - S2BIOPHYS", variable = "laie", breaks = c(0, 1,2,3,4,5),  ncol_facet = 4)
plot_grid(p1, p1_lc, p2, p2_lc, rel_widths = c(3, 4), ncol = 2)

plot_grid(p1, p2, ncol = 2)


#### Plot FAPAR
df_fapar <- df %>% dplyr::select(dplyr::matches('fapar'), 'NLCD', -dplyr::matches('std')) %>% 
  dplyr::rename(true = fapar, pred_sl2p = sl2p_fapar_mean, pred_specker = specker_fapar_mean)

metrics_sl2p <- calculate_metrics(df_fapar %>% dplyr::select(true, pred_sl2p) %>% dplyr::rename(pred = pred_sl2p), 'fapar')
metrics_specker <- calculate_metrics(df_fapar %>% dplyr::select(true, pred_specker) %>% dplyr::rename(pred = pred_specker), 'fapar')

p3 <- create_global_plot(df_fapar %>% dplyr::select(true, pred_sl2p) %>% dplyr::rename(pred = pred_sl2p), metrics_sl2p, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, y_label = "FAPAR - SL2P" , x_label = "FAPAR - RMs", variable = 'fapar', breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0))
p4 <- create_global_plot(df_fapar %>% dplyr::select(true, pred_specker) %>% dplyr::rename(pred = pred_specker), metrics_specker, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, y_label = "FAPAR - S2BIOPHYS" , x_label = "FAPAR - RMs", variable = 'fapar', breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0))

df_fapar_subset <- df_fapar %>% subset_lcs(top_n = 8)
p3_lc <- create_lc_plot(dplyr::select(df_fapar_subset, true, pred_sl2p, NLCD) %>% dplyr::rename(pred = pred_sl2p), xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, x_label = "FAPAR - RMs", y_label = "FAPAR - SL2P", variable = "fapar", breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),  ncol_facet = 4)
p4_lc <- create_lc_plot(dplyr::select(df_fapar_subset, true, pred_specker, NLCD) %>% dplyr::rename(pred = pred_specker), xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, x_label = "FAPAR - RMs", y_label = "FAPAR - S2BIOPHYS", variable = "fapar", breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),  ncol_facet = 4)
# plot_grid(p1, p1_lc, p2, p2_lc, rel_widths = c(3, 4), ncol = 2)

plot_grid(p3, p4, ncol = 2)

#### Plot FCOVER
df_fcover <- df %>% dplyr::select(dplyr::matches('fcover'), 'NLCD', -dplyr::matches('std')) %>% 
  dplyr::rename(true = fcover, pred_sl2p = sl2p_fcover_mean, pred_specker = specker_fcover_mean)

metrics_sl2p <- calculate_metrics(df_fcover %>% dplyr::select(true, pred_sl2p) %>% dplyr::rename(pred = pred_sl2p), 'fcover')
metrics_specker <- calculate_metrics(df_fcover %>% dplyr::select(true, pred_specker) %>% dplyr::rename(pred = pred_specker), 'fcover')

p5 <- create_global_plot(df_fcover %>% dplyr::select(true, pred_sl2p) %>% dplyr::rename(pred = pred_sl2p), metrics_sl2p, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, y_label = "FCOVER - SL2P" , x_label = "FCOVER - RMs", variable = 'fcover', breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0))
p6 <- create_global_plot(df_fcover %>% dplyr::select(true, pred_specker) %>% dplyr::rename(pred = pred_specker), metrics_specker, xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, y_label = "FCOVER - S2BIOPHYS" , x_label = "FCOVER - RMs", variable = 'fcover', breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0))

df_fcover_subset <- df_fcover %>% subset_lcs(top_n = 8)
p5_lc <- create_lc_plot(dplyr::select(df_fcover_subset, true, pred_sl2p, NLCD) %>% dplyr::rename(pred = pred_sl2p), xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, x_label = "FCOVER - RMs", y_label = "FCOVER - SL2P", variable = "fcover", breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),  ncol_facet = 4)
p6_lc <- create_lc_plot(dplyr::select(df_fcover_subset, true, pred_specker, NLCD) %>% dplyr::rename(pred = pred_specker), xlim = c(0, 1), ylim = c(0, 1), density_palette = palette, density_bandwidth = 0.1, x_label = "FCOVER - RMs", y_label = "FCOVER - S2BIOPHYS", variable = "fcover", breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),  ncol_facet = 4)
# plot_grid(p1, p1_lc, p2, p2_lc, rel_widths = c(3, 4), ncol = 2)

plot_grid(p5, p6, ncol = 2)



####### Figure 3: (Comparison S2BIOPHYS with SL2P) 757 x 563
plot_grid(p2, p4, p6, p1, p3, p5, ncol = 3, labels = c("A1", "A2", "A3", "B1", "B2", "B3"))


####### Supplementary Figure S5 # export 800 x 1280 # don't add for the moment (table is sufficient in my eyes)
plot_grid(p2_lc, p4_lc, p6_lc, ncol = 1, labels = c("A", "B", "C"))
