####
# Load data
####

library('tidyverse')
library('GGally')

leaf_foliar_traits_generated <- readr::read_csv('../data/validation_pipeline/output/foliar/neon_foliar_insitu_generated_copula_samples.csv')
leaf_foliar_traits_insitu <- readr::read_csv('../data/validation_pipeline/output/foliar/neon_foliar_insitu_merged_filtered.csv')

####
# Visualize in-situ samples and generated datapoints.
####

# plot distribution of sampled versus distribution

leaf_foliar_traits_generated <- leaf_foliar_traits_generated %>%
  mutate(Source = "generated") %>% dplyr::sample_n(size = nrow(leaf_foliar_traits_insitu))

leaf_foliar_traits_insitu <- leaf_foliar_traits_insitu %>%
  mutate(Source = "insitu")

combined_df <- bind_rows(leaf_foliar_traits_generated, leaf_foliar_traits_insitu)

# Custom function for upper panels: Contour plot
upper_fn <- function(data, mapping, ...){
  ggplot(data = data, mapping = mapping) +
    geom_density_2d(aes(color = Source)) +
    theme_minimal()
}

# Custom function for lower panels: Scatter plot
lower_fn <- function(data, mapping, ...){
  ggplot(data = data, mapping = mapping) +
    geom_point(aes(color = Source), alpha = 0.5, size = 0.8) +
    theme_minimal()
}


# Custom function for diagonal panels: Histogram
diag_fn <- function(data, mapping, ...){
  ggplot(data = data, mapping = mapping) +
    geom_histogram(bins = 20, aes(fill = Source), position = "identity", alpha = 0.5) +
    theme_minimal()
}

df_prepared <- combined_df %>% 
  dplyr::rename(
    `CHL [µg/cm²]` = chlorophyll_ab_mug_cm2,
    `EWT [cm]` = ewt_cm,
    `CAR [µg/cm²]` = carotenoid_mug_cm2,
    `LMA [g/cm²]` = leafMassPerArea_g_cm2
  )
# Using ggpairs to create the plot
p <- ggpairs(df_prepared,
             mapping = ggplot2::aes(color = Source),
             upper = list(continuous = wrap(upper_fn)),
             lower = list(continuous = wrap(lower_fn)),
             diag = list(continuous = wrap(diag_fn)),
             columns = c("CHL [µg/cm²]", "LMA [g/cm²]", "EWT [cm]", "CAR [µg/cm²]")
) 
# p <- ggpairs(df_prepared,
        # mapping = ggplot2::aes(color = 'Source'),
        # upper = list(continuous = wrap(upper_fn)),
        # lower = list(continuous = wrap(lower_fn)),
        # diag = list(continuous = wrap(diag_fn)),
        # # columns = c("chlorophyll_ab_mug_cm2", "leafMassPerArea_g_cm2", "ewt_cm", "carotenoid_mug_cm2")) +
        # columns = c("CHL", "LMA", "EWT", "CAR"))
  # ggtitle("Pairwise trait comparison: NEON-insitu (blue), Gaussian Copula Random (red)")
p %>% show()



# save plot
ggsave(file = 'output/foliar_traits_neon_pairwise.png', plot = p, width = 10, height = 8, device = 'png')
# ggsave(file = 'output/foliar_traits_neon_pairwise.svg', plot = p, width = 10, height = 8, device = 'svg')




