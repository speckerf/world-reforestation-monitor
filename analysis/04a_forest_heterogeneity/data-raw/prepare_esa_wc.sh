input_dir="data-raw/ESA_WorldCover_10m_2021_v200_60deg_macrotile_S90E120"

gdalbuildvrt \
  -srcnodata 0 \
  -vrtnodata 0 \
  data-raw/ESA_WorldCover_10m_2021_v200_tasmania.vrt \
  "$input_dir/ESA_WorldCover_10m_2021_v200_S42E141_Map.tif" \
  "$input_dir/ESA_WorldCover_10m_2021_v200_S42E144_Map.tif" \
  "$input_dir/ESA_WorldCover_10m_2021_v200_S42E147_Map.tif" \
  "$input_dir/ESA_WorldCover_10m_2021_v200_S45E144_Map.tif" \
  "$input_dir/ESA_WorldCover_10m_2021_v200_S45E147_Map.tif"

gdalwarp \
  data-raw/ESA_WorldCover_10m_2021_v200_tasmania.vrt \
  data/ESA_WorldCover_10m_2021_v200_tasmania_match_laie20.tif \
  -t_srs EPSG:32755 \
  -te_srs EPSG:32755 \
  -te 199020 5122160 628980 5626680 \
  -tr 20 20 \
  -r near \
  -srcnodata 0 \
  -dstnodata 0 \
  -ot Byte \
  -of COG \
  -co COMPRESS=DEFLATE \
  -co RESAMPLING=NEAREST \
  -co BLOCKSIZE=512 \
  -co BIGTIFF=IF_SAFER \
  -co NUM_THREADS=ALL_CPUS \
  -multi \
  -wo NUM_THREADS=ALL_CPUS
