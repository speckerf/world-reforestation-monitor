#!/usr/local/bin/R

####
# Load and filter Data
####

library('tidyverse')

path <- file.path('..', 'data', 'validation_pipeline', 'output', 'foliar', 'neon_foliar_insitu_merged.csv')
df <- readr::read_csv(path, col_select = c('plotID', 'lat', 'lon', 'date', 'chlorophyll_ab_mug_cm2', 'leafMassPerArea_g_cm2', 'ewt_cm', 'carotenoid_mug_cm2'))

# filter out all entries with CHL > 100 mug/cm2
# filter out all ewt_cm more than 0.1 cm
# filter out all leaf dry matter contents above 0.10

df_filtered = df %>%
  dplyr::filter(ewt_cm < 0.1 &
                  chlorophyll_ab_mug_cm2 < 100 &
                  leafMassPerArea_g_cm2 < 0.10 &
                  carotenoid_mug_cm2 < 30) %>%
  dplyr::select(c('chlorophyll_ab_mug_cm2', 'leafMassPerArea_g_cm2', 'ewt_cm', 'carotenoid_mug_cm2'))

readr::write_csv(df_filtered, file.path('..', 'data', 'validation_pipeline', 'output', 'foliar', 'neon_foliar_insitu_merged_filtered.csv'))

####
# Fit multivariate truncated normals / and sample from them
####

## fit gaussian copula:
library('copula')
library('truncnorm')

# Calculate the covariance matrix
cov_matrix <- cov(df_filtered %>% select(chlorophyll_ab_mug_cm2, leafMassPerArea_g_cm2, ewt_cm, carotenoid_mug_cm2))

# Convert covariance matrix to correlation matrix
cor_matrix <- cov2cor(cov_matrix)
param_vector <- cor_matrix[upper.tri(cor_matrix)]

# Define a Gaussian copula with the covariance matrix
normal_cop <- normalCopula(param = param_vector, dim = ncol(df_filtered), dispstr = 'un')

# with truncated normals:
margins <- rep('truncnorm', times = ncol(df_filtered))
params = lapply(colnames(df_filtered), function(i){
  list(a = 0, mean = mean(df_filtered[[i]]), sd = sd(df_filtered[[i]]))
})
# with standard normals instead
# margins <- rep('norm', times = ncol(df_filtered))
# params = lapply(colnames(df_filtered), function(i){
#   list(mean = mean(df_filtered[[i]]), sd = sd(df_filtered[[i]]))
# })


# Define the multivariate distribution with the copula and margins
multi_norm <- mvdc(copula = normal_cop, margins = margins, paramMargins = params)

leaf_foliar_traits_random <- rMvdc(n = 100000, mvdc = multi_norm)
colnames(leaf_foliar_traits_random) <- colnames(df_filtered)
leaf_foliar_traits_random <- leaf_foliar_traits_random %>% tibble::as_tibble()

readr::write_csv(leaf_foliar_traits_random, file = file.path('..', 'data', 'validation_pipeline', 'output', 'foliar', 'neon_foliar_insitu_generated_copula_samples.csv'))
