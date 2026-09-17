url='https://gfw-files.s3.amazonaws.com/plantations/SDPT_v2.1/sdpt_v21_v09152024_public.gdb.zip'
src="/vsizip//vsicurl/$url"

ogrinfo -ro "$src"


ogr2ogr -f GPKG sdpt_tasmania.gpkg \
  "$src" "aus_plant_v21" \
  -spat 143.5 -44.0 148.5 -39.5 \
  -spat_srs EPSG:4326 \
  -t_srs EPSG:4326 \
  -nln sdpt