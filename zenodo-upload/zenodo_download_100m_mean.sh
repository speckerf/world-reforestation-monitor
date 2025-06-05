#!/bin/bash
# This script downloads the base and children depositions - Uncomment relevant lines if only subset of data is desired 
# Total data size: ~ 550 GB (takes about 15 hours to Download at 10MB/s from Zenodo)

echo 'Downloading children depositions: 100m resolution (Uncomment the lines below to skip 100m resolution downloads)'

echo 'Downloading LAI depositions:'
wget https://zenodo.org/records/15053004/files/lai_rtm.mlp_mean_100m_s_20190101_20191231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053139/files/lai_rtm.mlp_mean_100m_s_20200101_20201231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053221/files/lai_rtm.mlp_mean_100m_s_20210101_20211231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053304/files/lai_rtm.mlp_mean_100m_s_20220101_20221231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053353/files/lai_rtm.mlp_mean_100m_s_20230101_20231231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053423/files/lai_rtm.mlp_mean_100m_s_20240101_20241231_go_epsg.4326_v01.tif

echo 'Downloading FCOVER depositions:'
wget https://zenodo.org/records/15053847/files/fcover_rtm.mlp_mean_100m_s_20190101_20191231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053875/files/fcover_rtm.mlp_mean_100m_s_20200101_20201231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053902/files/fcover_rtm.mlp_mean_100m_s_20210101_20211231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053932/files/fcover_rtm.mlp_mean_100m_s_20220101_20221231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053962/files/fcover_rtm.mlp_mean_100m_s_20230101_20231231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053989/files/fcover_rtm.mlp_mean_100m_s_20240101_20241231_go_epsg.4326_v01.tif

echo 'Downloading FAPAR depositions:'
wget https://zenodo.org/records/15053491/files/fapar_rtm.mlp_mean_100m_s_20190101_20191231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053549/files/fapar_rtm.mlp_mean_100m_s_20200101_20201231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053621/files/fapar_rtm.mlp_mean_100m_s_20210101_20211231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053689/files/fapar_rtm.mlp_mean_100m_s_20220101_20221231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053747/files/fapar_rtm.mlp_mean_100m_s_20230101_20231231_go_epsg.4326_v01.tif
wget https://zenodo.org/records/15053814/files/fapar_rtm.mlp_mean_100m_s_20240101_20241231_go_epsg.4326_v01.tif