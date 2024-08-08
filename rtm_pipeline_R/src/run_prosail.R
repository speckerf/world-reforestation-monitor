#!/usr/local/bin/R

# Load necessary libraries
library(jsonlite)
library(argparse)
library(prosail)
library(readr)
library(dplyr)
library(magrittr)


# Function to parse arguments
parse_args_wrapper <- function() {
  parser <- argparse::ArgumentParser(description='Run PROSAIL model and generate LUT')
  parser$add_argument('--input', required=TRUE, help='Path to input CSV file')
  parser$add_argument('--output', required=TRUE, help='Path to output CSV file')
  parser$add_argument('--add_noise', required=TRUE, help='Whether to add noise to reflectance')
  parser$add_argument('--noise_type', required=FALSE, help='Type of noise to add', default = NULL, type = 'character')
  parser$add_argument('--noise_args', required=FALSE, help='Additional arguments for noise', default='{}', type = 'character')
  parser$add_argument('--modify_rsoil', required = FALSE, help='Modify custom background soil reflectance', default=FALSE, type = 'logical')
  parser$add_argument('--ecoregion', required = FALSE, help='From which ecoregion to get background soil reflectances', default="All", type = 'character')
  parser$add_argument('--rsoil_insitu', required = FALSE, help='Use insitu samples to modify background soil reflectance', default=FALSE, type = 'logical')
  parser$add_argument('--rsoil_emit', required = FALSE, help='Use emit hyperspectral samples to modify background soil reflectance', default=FALSE, type = 'logical')
  parser$add_argument('--rsoil_insitu_fraction', required = FALSE, help='Fraction of insitu rsoil modifications', default=0.0, type = 'double')
  parser$add_argument('--rsoil_emit_fraction', required = FALSE, help='Fraction of emit rsoil modifications', default=0.0, type = 'double')
  return(parser$parse_args())
}

# Simulate command line arguments if running interactively
if (interactive()) {
  input_file <- tempfile(fileext = ".csv")
  output_file <- tempfile(fileext = ".csv")

  lut <- prosail::get_atbd_LUT_input()
  readr::write_csv(lut, input_file)
  
  args <- list(
    input = input_file,
    output = output_file,
    add_noise = TRUE,
    noise_type = "atbd",
    # noise_args = '{"AdditiveNoise": 0.05886511830424981, "MultiplicativeNoise": 0.4128726726941848}',
    noise_args = '{}',
    modify_rsoil = TRUE,
    ecoregion = "All",
    rsoil_insitu = FALSE,
    rsoil_emit = TRUE,
    rsoil_insitu_fraction = 0.0,
    rsoil_emit_fraction = 0.5
  )
} else {
  args <- parse_args_wrapper()
}
path_to_root <- "/Users/felix/Projects/OEMC/world-reforestation-monitor"


# Main function to run PROSAIL model
run_prosail <- function(args) {
  # browser()
  print(paste("Running PROSAIL model with input:", args$input))
  # browser()  # Debugging point
  
  # Load the input data
  input_prosail <- readr::read_csv(args$input)
  
  # load table with all ecoregions and their biome number
  eco_biom_table <- readr::read_csv(file = file.path(path_to_root, "data", "misc", "ecoregion_biome_table.csv")) %>% dplyr::select(c('ECO_ID', 'BIOME_NUM'))
  
  # Define SensorName and get SRF
  SensorName <- "S2"
  SRF <- prosail::GetRadiometry(SensorName = SensorName)
  
  if (args$modify_rsoil) {
    print(paste0("args$modify_rsoil ", args$modify_rsoil))
    print(paste0("args$rsoil_insitu ", args$rsoil_insitu))
    print(paste0("args$rsoil_emit " , args$rsoil_emit))
    assertthat::assert_that(args$rsoil_insitu | args$rsoil_emit)
    assertthat::assert_that(!is.null(args$ecoregion))
    if (args$rsoil_insitu && !args$rsoil_emit) {
      assertthat::assert_that(args$rsoil_insitu_fraction > 0)
      if(args$ecoregion == "All"){
        path_all = file.path(path_to_root, "data", "rtm_pipeline", "output", "insitu_soil_database", "insitu_soil_spectra_hyperspectral_small_sample.csv")
        rsoil_insitu <- readr::read_csv(path_all, col_select = dplyr::matches('scan_visnir'))
      } else{
        path_eco = file.path(path_to_root, "data", "rtm_pipeline", "output", "insitu_soil_database", paste0("insitu_soil_spectra_hyperspectral_eco_", args$ecoregion, ".csv"))
        if(file.exists(path_eco)){
          rsoil_insitu <- readr::read_csv(path_eco, col_select = dplyr::matches('scan_visnir'))
        } else{
          message(paste0("No ecoregion-level insitu soil samples in database, taking biome-level soil samples instead"))
          biome_num <- dplyr::filter(eco_biom_table, ECO_ID == args$ecoregion)[['BIOME_NUM']] %>% unique()
          message(paste0('Biome Number', biome_num))
          path_biome = file.path(path_to_root, "data", "rtm_pipeline", "output", "insitu_soil_database", paste0("insitu_soil_spectra_hyperspectral_biome_", as.integer(biome_num), ".csv"))
          rsoil_insitu <- readr::read_csv(path_biome, col_select = dplyr::matches('scan_visnir'))
        }
      }
    } else if (args$rsoil_emit && !args$rsoil_insitu) {
      assertthat::assert_that(args$rsoil_emit_fraction > 0)
      if(args$ecoregion == "All"){
        path_all = file.path(path_to_root, "data", "rtm_pipeline", "output", "emit_hyperspectral", "global-baresoil-random-points-small_sample_hyperspectral.csv")
        rsoil_emit <- readr::read_csv(path_all, col_select = dplyr::matches("\\d{3,}"))
      } else{
        path_eco = file.path(path_to_root, "data", "rtm_pipeline", "output", "emit_hyperspectral", "point_data", paste0("global-baresoil-random-points-eco_", args$ecoregion, "_hyperspectral.csv"))
        if(file.exists(path_eco)){
          rsoil_emit <- readr::read_csv(path_eco, col_select = dplyr::matches("\\d{3,}"))
        } else{
          message(paste0("No ecoregion-level emit soil reflectances in database, taking biome-level soil samples instead"))
          biome_num <- dplyr::filter(eco_biom_table, ECO_ID == args$ecoregion)[['BIOME_NUM']] %>% unique()
          path_biome = file.path(path_to_root, "data", "rtm_pipeline", "output", "emit_hyperspectral", "point_data", paste0("global-baresoil-random-points-biome_", as.integer(biome_num), "_hyperspectral.csv"))
          if(file.exists(path_biome)){
            rsoil_emit <- readr::read_csv(path_biome, col_select = dplyr::matches("\\d{3,}"))  
          } else{
            message(paste0("No biome-level emit soil reflectances in database, using global random sample instead"))
            path_all = file.path(path_to_root, "data", "rtm_pipeline", "output", "insitu_soil_database", "insitu_soil_spectra_hyperspectral_small_sample.csv")
            rsoil_emit <- readr::read_csv(path_all, col_select = dplyr::matches("\\d{3,}"))  
          }
        }
      }
    } else if (args$rsoil_emit && args$rsoil_insitu) {
      assertthat::assert_that(args$rsoil_insitu_fraction > 0)
      assertthat::assert_that(args$rsoil_emit_fraction > 0)
      stop("Not implemented yet both rsoil from emit and insitu observations...")
      # should up or downsample both rsoil sources somehow. 
    }
    
    
    if(exists('rsoil_insitu')){
      custom_rsoil <- rsoil_insitu %>%
        t() %>% tibble::as_tibble() %>%
        dplyr::mutate(lambda = seq(400, 2500, 1)) %>%
        dplyr::relocate(lambda)
    } else if(exists('rsoil_emit')){
      custom_rsoil <- rsoil_emit %>%
        t() %>% tibble::as_tibble() %>%
        dplyr::mutate(lambda = seq(400, 2500, 1)) %>%
        dplyr::relocate(lambda)
    }
  }
  
  # Generate LUT using PROSAIL
  if (!args$modify_rsoil) {
    message("Generating LUT...")
    res <- prosail::Generate_LUT_PROSAIL(SAILversion = '4SAIL', InputPROSAIL = input_prosail,
                                         SpecPROSPECT = prosail::SpecPROSPECT_FullRange, SpecSOIL = prosail::SpecSOIL,
                                         SpecATM = prosail::SpecATM)
  } else{
    message("Generating LUT with modified rsoil...")
    res <- prosail::Generate_LUT_PROSAIL(SAILversion = '4SAIL', InputPROSAIL = input_prosail,
                                         SpecPROSPECT = prosail::SpecPROSPECT_FullRange, SpecSOIL = prosail::SpecSOIL,
                                         SpecATM = prosail::SpecATM, useCustomSoil = args$modify_rsoil, customSoilSpec = custom_rsoil, customSoilProba = args$rsoil_insitu_fraction)
  }
  
  BRF_LUT_1nm <- res$BRF
  input_prosail$fCover <- res$fCover
  input_prosail$fAPAR <- res$fAPAR
  input_prosail$albedo <- res$albedo
  
  BRF_LUT <- prosail::applySensorCharacteristics(wvl = prosail::SpecPROSPECT_FullRange$lambda,
                                                 SRF = SRF,
                                                 InRefl = BRF_LUT_1nm)
  # Identify spectral bands in LUT
  rownames(BRF_LUT) <- SRF$Spectral_Bands
  
  # Apply noise if specified
  if (args$add_noise == TRUE) {
    message("Adding noise...")
    assertthat::assert_that(!is.null(args$noise_type))
    if (args$noise_type == "atbd") {
      message("Adding atbd noise...")
      BRF_LUT_NOISE <- prosail::apply_noise_atbd(BRF_LUT)
    } else if (args$noise_type == "addmulti") {
      message("Adding additive / multiplicative noise...")
      noise_args <- fromJSON(args$noise_args)
      message("With params: Additive_Noise: ", noise_args$AdditiveNoise, " Multiplicative Noise: ", noise_args$MultiplicativeNoise)
      BRF_LUT_NOISE <- prosail::apply_noise_AddMult(BRF_LUT,
                                                    AdditiveNoise = noise_args$AdditiveNoise,
                                                    MultiplicativeNoise = noise_args$MultiplicativeNoise)
    } else {
      stop("Unsupported noise type specified.")
    }
    LUT <- cbind(input_prosail, t(BRF_LUT_NOISE))
  } else {
    LUT <- cbind(input_prosail, t(BRF_LUT))
  }
  
  # Write the final LUT to disk
  write.csv(LUT, args$output, row.names = FALSE)
}

# Run the main function with parsed arguments
run_prosail(args)
message('Prosail forward pass successfull')


