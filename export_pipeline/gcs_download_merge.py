import glob
import os
import subprocess

import ee
from google.cloud import storage
from loguru import logger

from config.config import get_config

ee.Initialize()
CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


def sync_gcs_to_local(temp_gcs_folder, local_folder, filename):
    """Syncs a GCS bucket to a local folder using gsutil rsync"""
    # create dir filename if not exists, in local folder
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    if not os.path.exists(f"{local_folder}/{filename}"):
        os.makedirs(f"{local_folder}/{filename}")

    # caution!! using option -d: delete files in local folder that are not in GCS
    cmd = f"gcloud storage rsync --delete-unmatched-destination-objects -r --dry-run {temp_gcs_folder} {local_folder}"
    # cmd = f"gcloud storage rsync -r {temp_gcs_folder} {local_folder}"
    subprocess.run(cmd, shell=True)


def main():
    filename = "lai_rtm.mlp_mean_100m_s_20190101_20191231_go_epsg.4326_v10"
    trait_name = "lai_mean"
    assert filename.startswith(trait_name.split("_")[0])
    assert trait_name.split("_")[1] in filename
    bucket_name = CONFIG_GEE_PIPELINE["GCLOUD_FOLDERS"]["BUCKET"]
    temp_gcs_folder = f"gs://{bucket_name}/{CONFIG_GEE_PIPELINE['GCLOUD_FOLDERS']['EXPORT_FOLDER_INTERMEDIATE']}"
    temp_local_folder = CONFIG_GEE_PIPELINE["LOCAL_FOLDERS"][
        "EXPORT_FOLDER_INTERMEDIATE"
    ]
    output_local_folder = CONFIG_GEE_PIPELINE["LOCAL_FOLDERS"]["EXPORT_FOLDER_FINAL"]

    sync_gcs_to_local(temp_gcs_folder, temp_local_folder, filename)

    # list all files
    full_paths = glob.glob(f"{temp_local_folder}/{filename}/*.tif")

    # check that they all contain filename substring
    for file in full_paths:
        assert filename in file

    # merge all files into a single file using gdal / directly convert to COG
    logger.info(f"Merging {len(full_paths)} files using gdal_merge.py...")
    output_file = filename
    cmd = f"{CONFIG_GEE_PIPELINE['CONDA_PATH']}/bin/gdal_merge.py -co BIGTIFF=IF_SAFER -co COMPRESS=DEFLATE -o {os.path.join(temp_local_folder, output_file)}.tif {' '.join(full_paths)}"
    # cmd = f'gdal_merge.py -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER -o {output_folder}/{output_file} {" ".join(full_paths)}'
    subprocess.run(cmd, shell=True, check=True)

    ## edit offset scale metadata
    scale = (
        1 / CONFIG_GEE_PIPELINE["INT16_SCALING"][trait_name]
    )  # one over the scaling factor
    offset = 0
    cmd = f"gdal_edit.py -scale {scale} -offset {offset} -approx_stats -a_nodata 0 {temp_local_folder}/{output_file}"
    subprocess.run(cmd, shell=True, check=True)

    # convert to COG
    logger.info(f"Converting {output_file} to COG format...")
    cmd = f"gdal_translate -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER {temp_local_folder}/{output_file} {output_local_folder}/{output_file}"
    subprocess.run(cmd, shell=True, check=True)

    # delete the non-cog file
    logger.info(f"Deleting {output_file}...")
    cmd = f"rm {temp_local_folder}/{output_file}"
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
