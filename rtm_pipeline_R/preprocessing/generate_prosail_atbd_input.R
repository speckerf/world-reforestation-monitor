library(prosail)
library(dplyr)
library(magrittr)

n_samples <- 100000
LUT <- prosail::get_atbd_LUT_input(nbSamples = n_samples) %>% dplyr::select(-c('tto', 'tts', 'psi'))

output_path <- file.path('..', 'data', 'rtm_pipeline', 'input', 'prosail_atbd', 'atbd_inputs.csv')



# Write the final LUT to disk
write.csv(LUT, output_path, row.names = FALSE)


