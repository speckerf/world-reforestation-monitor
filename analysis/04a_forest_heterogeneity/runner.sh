
python analysis/01_create_forest_window_sampling_mask.py \
  --jrc data/JRC_GFT2020_V1_tasmania_match_laie20.tif \
  --worldcover data/ESA_WorldCover_10m_2021_v200_tasmania_match_laie20.tif \
  --osm data/osm_roads_buildings_20m.tif \
  --output data/forest_window_sampling_mask_500m.tif

python analysis/02_match_forest_window_centres.py \
  --mask data/forest_window_sampling_mask_500m.tif \
  --output results/forest_window_matched_pairs.gpkg \
  --max-distance-m 5000 \
  --exclusion-distance-m 1000 \
  --iterations 100 \
  --seed 42

python analysis/03_analyse_matched_pair_laie_heterogeneity.py \
  --pairs results/forest_window_matched_pairs.gpkg \
  --laie data/laie_tasmania/tasmania_laie_2020_20m.tif \
  --output results/matched_pair_laie_heterogeneity.csv


python analysis/04_summarise_laie_heterogeneity_effects.py \
  --input results/matched_pair_laie_heterogeneity.csv \
  --summary results/laie_heterogeneity_effect_summary.csv \
  --figure figures/laie_heterogeneity_paired_differences.png \
  --control-type naturally_regenerating \
  --bootstrap-samples 20000 \
  --seed 42

python analysis/05_plot_forest_heterogeneity_figure.py \
  --laie data/laie_tasmania/tasmania_laie_2020_20m.tif \
  --jrc data/JRC_GFT2020_V1_tasmania_match_laie20.tif \
  --pairs results/forest_window_matched_pairs.gpkg \
  --metrics results/matched_pair_laie_heterogeneity.csv \
  --output figures/forest_heterogeneity_comparison.png \
  --bootstrap-samples 20000 \
  --seed 42
