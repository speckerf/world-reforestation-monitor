gdalbuildvrt \
  jrc_gft/JRC_GFT2020_V1_tasmania.vrt \
  jrc_gft/JRC_GFT2020_V1_S30_E140.tif \
  jrc_gft/JRC_GFT2020_V1_S40_E140.tif



gdal_translate \
  jrc_gft/JRC_GFT2020_V1_tasmania.vrt \
  jrc_gft/JRC_GFT2020_V1_tasmania.tif \
  -projwin 143.5 -39.5 148.5 -44.0 \
  -co COMPRESS=LZW \
  -co TILED=YES \
  -co BIGTIFF=YES \
  -co NUM_THREADS=ALL_CPUS


gdalwarp \
  jrc_gft/JRC_GFT2020_V1_tasmania.tif \
  jrc_gft/JRC_GFT2020_V1_tasmania_match_laie20.tif \
  -t_srs EPSG:32755 \
  -te_srs EPSG:32755 \
  -te 199020 5122160 628980 5626680 \
  -tr 20 20 \
  -r near \
  -srcnodata 0 \
  -dstnodata 0 \
  -co TILED=YES \
  -co COMPRESS=LZW \
  -co BIGTIFF=YES \
  -co NUM_THREADS=ALL_CPUS
