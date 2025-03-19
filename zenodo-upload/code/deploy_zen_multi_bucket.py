import glob
import os
from itertools import product
from pprint import pprint
from typing import Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from zen import LocalFiles, Zenodo

FINAL_DEPOSITION = True
# parse zenodo token from auth.zenodo_sandbox.txt
if not FINAL_DEPOSITION:
    with open("auth/zenodo_sandbox.txt", "r") as f:
        ZENODO_ACCESS_TOKEN = f.read().strip()
        DEPOSITION_METADATA_PATH = (
            "zenodo-upload/depositions/test/test-deposition-base.json"
        )
else:
    # raise ValueError("Final deposition not implemented yet.")
    with open("auth/zenodo.txt", "r") as f:
        ZENODO_ACCESS_TOKEN = f.read().strip()
        DEPOSITION_METADATA_PATH = (
            "zenodo-upload/depositions/deploy/deploy-deposition-base.json"
        )

def grep_filenames(
    folder: str = ".", suffix: str = ".tif", contains: Union[str, list] = None
):
    if contains:
        files = glob.glob(f"{folder}/**/*{suffix}", recursive=True)
        if isinstance(contains, str):
            return [f for f in files if contains in f]
        if isinstance(contains, list):
            return [f for f in files if all(c in f for c in contains)]
    else:
        return glob.glob(f"{folder}/**/*{suffix}", recursive=True)


def get_preview_file(tif_path: str, folder_previews: str) -> str: 
    # find matching preview file in folder_previews
    preview_files = grep_filenames(folder=folder_previews, suffix=".png", contains=os.path.basename(tif_path).replace(".tif", ""))
    if len(preview_files) == 1:
        return preview_files[0]
    else:
        raise ValueError(f"Found {len(preview_files)} preview files for {tif_path}. Expected 1.")


def main():
    """
    Create base deposition with all 1000m resolution files.
    """
    if FINAL_DEPOSITION:
        zen = Zenodo(url=Zenodo.url, token=ZENODO_ACCESS_TOKEN)
    else:
        zen = Zenodo(url=Zenodo.sandbox_url, token=ZENODO_ACCESS_TOKEN)

    # DATA_FOLDER_1000m = "data-local/results_1000m/"
    DATA_FOLDER_1000m = "/Volumes/OEMC/world-reforestation-monitor/results_1000m/"
    # DATA_FOLDER_100m = "data-local/results_1000m/"
    DATA_FOLDER_100m = "/Volumes/OEMC/world-reforestation-monitor/results_100m/"
    doi_prefix = "https://doi.org/"

    creators_list = [
        {"name": "Felix Specker", "affiliation": "Institute of Integrative Biology, Department of Environmental Systems Science, ETH Zurich, Switzerland", "orcid": "0000-0002-9398-9975"},
        {"name": "Anna K. Schweiger", "affiliation": "Montana State University, Department of Land Resources and Environmental Sciences, Bozeman, MT, United States", "orcid": "0000-0002-5567-4200"},
        {"name": "Jean-Baptiste Féret", "affiliation": "TETIS, INRAE, AgroParisTech, CIRAD, CNRS, Université Montpellier, Montpellier, France", "orcid": "0000-0002-0151-1334"},
        {"name": "Thomas Lauber", "affiliation": "Institute of Integrative Biology, Department of Environmental Systems Science, ETH Zurich, Switzerland", "orcid": "0000-0002-3118-432X"},
        {"name": "AUTHORS GBOV", "affiliation": "Misc"},
        {"name": "Thomas W. Crowther", "affiliation": "Institute of Integrative Biology, Department of Environmental Systems Science, ETH Zurich, Switzerland", "orcid": "0000-0001-5674-8913"},
        {"name": "Johan van den Hoogen", "affiliation": "Institute of Integrative Biology, Department of Environmental Systems Science, ETH Zurich, Switzerland", "orcid": "0000-0001-6624-8461"},
    ]
    keywords_list = [
        "Vegetation traits",
        "Biodiversity",
        "Sentinel-2",
        "LAI",
        "FAPAR",
        "FCOVER",
        "PROSAIL",
    ]


    local_file_paths = grep_filenames(
        folder=DATA_FOLDER_1000m,
        suffix=".tif",
        contains=["_1000m"]
    )
    selected_preview_files = [f for f in local_file_paths if "1000m_s_20200101_20201231" in f]
    local_file_paths_selected_previews = list(map(lambda x: get_preview_file(x, "/Volumes/OEMC/world-reforestation-monitor/previews"), selected_preview_files))

    base_json_path = DEPOSITION_METADATA_PATH
    base_ds = LocalFiles(
        sorted([*local_file_paths, *local_file_paths_selected_previews], reverse=False, key=lambda x: os.path.basename(x)),
        dataset_path=base_json_path,
    )

    # load description.html from templates
    with open("zenodo-upload/templates/description.html", "r") as f:
        base_description = f.read()

    base_metadata = {
        "title": f"Advancing Ecosystem Monitoring: Global Annual Maps of Biophysical Vegetation Properties (LAI, FAPAR, FCOVER) for 2019-2024",
        "description": base_description,
        "upload_type": "dataset",
        "publication_type": "article",
        "keywords": keywords_list,
        "license": "cc-by-4.0",
        "creators": creators_list,
        "grants": [
            {"id": "101059548"},
        ],
        "communities": [
            {"identifier": "oemc-project"},
        ],
    }

    base_ds.set_deposition(api=zen, create_if_not_exists=True, metadata=base_metadata)
    base_ds.save()

    # reopen the dataset / deposition
    base_ds = LocalFiles.from_file(base_json_path)
    base_ds.set_deposition(api=zen, create_if_not_exists=False, metadata=base_metadata)
    base_ds_doi = base_ds.deposition.doi

    dep = base_ds.deposition

    # upload the files
    base_ds.update_metadata()
    base_ds.upload()

    """
    Create Code/Data deposition, containing the code and data used for model training/prediction
    """
    data_file_zipped = "data.tar.gz"
    code_file_zipped = "world-reforestation-monitor-main.zip"

    code_data_description = """
    <p>This deposition contains the code and data used for training and prediction of the PROSAIL model to predict LAI, FAPAR, and FCOVER.</p>

    <p>The code is available on GitHub: 
        <a href="https://github.com/speckerf/world-reforestation-monitor" target="_blank">
            world-reforestation-monitor
        </a>
    </p>

    <p>To use this deposition, please follow these steps:</p>
    <ol>
        <li>Download both the code archive (<code>world-reforestation-monitor-main.zip</code>) and the data archive (<code>data.tar.gz</code>).</li>
        <li>Unpack the code archive first, which will create a folder named <code>world-reforestation-monitor</code>.</li>
        <li>Unpack the data archive inside the <code>world-reforestation-monitor</code> folder. This will ensure the data is placed correctly for use with the provided scripts.</li>
    </ol>

    <p>Please refer to the base deposition for the uploaded generated maps and a complete description: 
        <a href="{doi_prefix}{doi}" target="_blank">{doi}</a>
    </p>
    """.format(doi_prefix=doi_prefix, doi=base_ds_doi)

    code_data_metadata = {
        "title": "Advancing Ecosystem Monitoring: Code and Data for Model Training and Prediction",
        "description": code_data_description,
        "upload_type": "software",
        "keywords": keywords_list,
        "license": "cc-by-4.0",
        "creators": creators_list,
        "grants": [
            {"id": "101059548"},
        ],
        "communities": [
            {"identifier": "oemc-project"},
        ],
    }

    code_data_ds = LocalFiles(
        [data_file_zipped, code_file_zipped],
        dataset_path=DEPOSITION_METADATA_PATH.replace(
            ".json", "-code-data.json"
        ),
    )

    code_data_ds.set_deposition(
        api=zen,
        create_if_not_exists=True,
        metadata=code_data_metadata,
    )

    # add related identifiers to base deposition: isSupplementTo
    code_data_ds.deposition.metadata.related_identifiers.add(
        **{
            "relation": "isSupplementTo",
            "identifier": base_ds.deposition.doi,
            "resource_type": "dataset",
        }
    )

    code_data_ds.save()
    code_data_ds.upload()
    code_data_ds.update_metadata()

    # update base deposition with related identifiers to code/data deposition
    base_ds.deposition.metadata.related_identifiers.add(
        **{
            "relation": "isSupplementedBy",
            "identifier": code_data_ds.deposition.doi,
            "resource_type": "software",
        }
    )
    # update to description: the link to DOI of code/data deposition
    base_ds.deposition.metadata.description = base_ds.deposition.metadata.description.replace(
        "ADD_DOI_CODE_REPO", f'<a href="{doi_prefix}{code_data_ds.deposition.doi}" target="_blank">{code_data_ds.deposition.doi}</a>'
    )
    base_ds.update_metadata()

    """
    Create children depositions with 100m resolution files.
    """

    # now create children depositions: assert each is less than 50 GB
    traits = ["LAI", "FAPAR", "FCOVER"]
    years = range(2019, 2025)
    children_ds = {}
    children_deps = {}
    for i, (trait, year) in enumerate(product(traits, years)):
        #######
        ## Deposit mean maps
        #######
        child_metadata = base_metadata.copy()
        # child_metadata["description"] = f"{trait} deposition with {len(base_ds)} files"
        child_metadata["description"] = (
            f'<h3>Subdataset: {trait} {year} [mean] </h3> Mean {trait} predictions for {year} at 100m resolution. See base depositions for more information: <a href="{doi_prefix}{base_ds.deposition.doi}" target="_blank">{base_ds_doi}</a>'
        )
        # child_metadata["keywords"].append(trait)
        child_metadata["related_identifiers"] = [
            {"relation": "isPartOf", "identifier": base_ds.deposition.doi}
        ]

        local_file_paths_children = grep_filenames(
            folder=DATA_FOLDER_100m,
            suffix=".tif",
            contains=[trait.lower(), "mean", str(year)],
        )
        local_file_paths_children_previews = list(map(lambda x: get_preview_file(x, "/Volumes/OEMC/world-reforestation-monitor/previews"), local_file_paths_children))

        current_ds = LocalFiles(
            sorted([*local_file_paths_children, *local_file_paths_children_previews], reverse=True, key=lambda x: os.path.basename(x)),
            dataset_path=DEPOSITION_METADATA_PATH.replace(
                ".json", f"-{trait.lower()}-{year}-mean.json"
            ),
        )

        current_ds.set_deposition(
            api=zen,
            create_if_not_exists=True,
            metadata=child_metadata,
        )
        current_ds.save()
        current_dep = current_ds.deposition
        # current_dep.metadata.description
        current_ds.upload()
        current_ds.update_metadata()
        pprint(
            f"{current_ds.dataset_path} created with {len(current_ds)} files. Size: {current_ds.storage_size * 1e-9:.2f} GB"
        )
        assert current_ds.storage_size < 50e9, "Dataset too large"

        children_ds[f"{trait}-{year}-mean"] = current_ds
        children_deps[f"{trait}-{year}-mean"] = current_dep

        #######
        ## Deposit std and count maps
        #######
        child_metadata = base_metadata.copy()
        # child_metadata["description"] = f"{trait} deposition with {len(base_ds)} files"
        child_metadata["description"] = (
            f'<h3>Subdataset: {trait} {year} [std / count]</h3> Standard deviation and count (number of observations) for {trait} predictions for {year} at 100m resolution. See base depositions for more information: <a href="{base_ds.deposition.doi}" target="_blank">{base_ds_doi}</a>'
        )
        child_metadata["related_identifiers"] = [
            {
                "relation": "isPartOf",
                "identifier": base_ds.deposition.doi,
                "resource_type": "dataset",
            }
        ]

        std_files = grep_filenames(
            folder=DATA_FOLDER_100m,
            suffix=".tif",
            contains=[trait.lower(), "std", str(year)],
        )
        count_files = grep_filenames(
            folder=DATA_FOLDER_100m,
            suffix=".tif",
            contains=[trait.lower(), "count", str(year)],
        )
        local_file_paths_std_count = std_files + count_files
        local_file_paths_children_previews = list(map(lambda x: get_preview_file(x, "/Volumes/OEMC/world-reforestation-monitor/previews"), local_file_paths_std_count))


        current_ds = LocalFiles(
            sorted([*local_file_paths_std_count, *local_file_paths_children_previews], reverse=True, key=lambda x: os.path.basename(x)),
            # template=base_template_100m,
            dataset_path=DEPOSITION_METADATA_PATH.replace(
                ".json", f"-{trait.lower()}-{year}-std-count.json"
            ),
        )

        current_ds.set_deposition(
            api=zen,
            create_if_not_exists=True,
            metadata=child_metadata,
        )
        current_ds.save()
        current_dep = current_ds.deposition
        # current_dep.metadata.description
        current_ds.upload()
        current_ds.update_metadata()
        pprint(
            f"{current_ds.dataset_path} created with {len(current_ds)} files. Size: {current_ds.storage_size * 1e-9:.2f} GB"
        )
        assert current_ds.storage_size < 50e9, "Dataset too large"

        children_ds[f"{trait}-{year}-stdcount"] = current_ds
        children_deps[f"{trait}-{year}-stdcount"] = current_dep

    # update base deposition with related identifiers
    for deposition_key, ds in children_ds.items():
        base_ds.deposition.metadata.related_identifiers.add(
            **{
                "relation": "hasPart",
                "identifier": ds.deposition.doi,
                "resource_type": "dataset",
            }
        )

    base_ds.deposition.update()

    # link children year to year: continues, isContinuedBy
    for deposition_key, dep in children_deps.items():
        trait, year, var = deposition_key.split("-")
        previous_year = int(year) - 1
        previous_dep_key = f"{trait}-{previous_year}-{var}"
        next_year = int(year) + 1
        next_dep_key = f"{trait}-{next_year}-var"

        if previous_dep_key in children_deps:
            dep.metadata.related_identifiers.add(
                **{
                    "relation": "continues",
                    "identifier": children_deps[previous_dep_key].doi,
                    "resource_type": "dataset",
                }
            )
        if next_dep_key in children_deps:
            dep.metadata.related_identifiers.add(
                **{
                    "relation": "isContinuedBy",
                    "identifier": children_deps[next_dep_key].doi,
                    "resource_type": "dataset",
                }
            )
        dep.update()


    #####
    # Updates Links in Base Depositions Metadata
    #####
    # update Base depositions description: add section with links to children / replace line: <p >ADD_RELATED_DOI_LINKS</p> with: 
    base_dep_description = base_ds.deposition.metadata.description
    doi_prefix = "https://doi.org/"
    html_string = "<h3>Related DOI Links</h3><ul>"

    for trait in traits:
        html_string += f"<li><strong>{trait}</strong><ul>"
        
        for var in ["mean", "stdcount"]:
            html_string += f"<li>{'Mean' if var == 'mean' else 'Std / Count'}<ul>"
            
            for year in years:
                dep_key = f"{trait}-{year}-{var}"
                dep = children_deps.get(dep_key)  # Avoid KeyError with .get()
                
                if dep:  # Ensure dep exists before using its attributes
                    html_string += f'<li><a href="{doi_prefix}{dep.doi}" target="_blank">{year}</a></li>'
            
            html_string += "</ul></li>"  # Close variable ul and li

        html_string += "</ul></li>"  # Close trait ul and li

    html_string += "</ul>"  # Close the outer ul

    base_dep_description_updated = base_dep_description.replace(
        "ADD_RELATED_DOI_LINKS",
        html_string
    )
    base_ds.deposition.metadata.description = base_dep_description_updated
    base_ds.deposition.update()

    # # cleanup - delete all depositions
    # base_ds.deposition.discard()
    # for ds in children_ds.values():
    #     ds.deposition.discard()

    # # delete all files in zenodo-upload/depositions/test/.json
    # for f in glob.glob("zenodo-upload/depositions/test/*.json"):
    #     os.remove(f)

def get_cms(trait, variable):
    assert trait in ["lai", "fapar", "fcover"]
    assert variable in ["mean", "std", "count"]

    if trait == 'lai':
        if variable == 'mean':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=5), cmap=mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#fffdcd', '#e1cd73', '#aaac20', '#5f920c', '#187328', '#144b2a', '#172313'], N=256))
        elif variable == 'std':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=2), cmap=mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#440154', '#433982', '#30678D', '#218F8B', '#36B677', '#8ED542', '#FDE725'], N=256))
        elif variable == 'count':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=30), cmap='viridis')
    elif trait == 'fapar':
        if variable == 'mean':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#ffffdd', '#e6ad12', '#c53859', '#3a26a1', '#000000'], N=256))
        elif variable == 'std':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=0.3), cmap=mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#440154', '#433982', '#30678D', '#218F8B', '#36B677', '#8ED542', '#FDE725'], N=256))
        elif variable == 'count':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=30), cmap='viridis')
    elif trait == 'fcover':
        if variable == 'mean':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45', '#00441b'], N=256))
        elif variable == 'std':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=0.3), cmap=mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#440154', '#433982', '#30678D', '#218F8B', '#36B677', '#8ED542', '#FDE725'], N=256))
        elif variable == 'count':
            return cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=30), cmap='viridis')
    else:
        raise ValueError(f"Trait {trait} not recognized.")


def generate_low_res_preview(tif_path):
    path = os.path.basename(tif_path)
    trait, variable, year = path.split('_')[0], path.split('_')[2], path.split('_')[5][:4]
    print(f"Generating low resolution preview for {trait} {variable} {year} file: {tif_path}")
    
    # Open the COG file
    with rasterio.open(tif_path, dtype=np.int16) as dataset:
        no_data_value = int(dataset.nodata)
        scaling_factor = dataset.scales[0]
        print(f"No data value: {no_data_value}, Scaling factor: {scaling_factor}")

        # Ensure the file has overviews
        if not dataset.overviews(1):
            raise ValueError("No overviews found in the provided COG file.")

        # Select the 2nd last overview level
        overview_levels = dataset.overviews(1)
        if len(overview_levels) < 2:
            raise ValueError("Not enough overviews found in the provided COG file.")
        overview_idx = -2
        ovr_factor = overview_levels[overview_idx]

        # Read the data at the selected overview level
        data = dataset.read(
            1, out_shape=(dataset.height // ovr_factor, dataset.width // ovr_factor)
        )

        if no_data_value:
            data = np.where(data == no_data_value, np.nan, data)


        if scaling_factor:
            data *= scaling_factor

        cm_scalar_mappable = get_cms(trait, variable)

        # Plot the global preview with colorbar below
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data, cmap=cm_scalar_mappable.cmap, interpolation="nearest")
        ax.set_title(f"{trait.capitalize()} {variable.capitalize()} {year}")
        ax.axis("off")

        # Create colorbar below
        cbar = fig.colorbar(
            cm_scalar_mappable,
            ax=ax,
            orientation="horizontal",
            pad=0.1,
            fraction=0.05,
        )
        cbar.set_label(f"{trait.capitalize()} {variable.capitalize()}")

        preview_filename = path.replace(".tif", "_preview.png")
        # add prefix for auto-sorting in zenodo: 00_ for mean, 01_ for std, 02_ for count
        if variable == 'mean':
            preview_filename = '00_' + preview_filename
        elif variable == 'std':
            preview_filename = '01_' + preview_filename
        elif variable == 'count':
            preview_filename = '02_' + preview_filename
    
        # get second dir in tif_path: results_1000m
        # save_dir = os.path.join(tif_path.split('/')[0], 'previews')
        if 'results_1000m' in tif_path:
            save_dir = os.path.split(tif_path)[0].replace('results_1000m', 'previews')
        elif 'results_100m' in tif_path:
            save_dir = os.path.split(tif_path)[0].replace('results_100m', 'previews')
        else:
            raise ValueError(f"Path {tif_path} not recognized.")
        save_filename = os.path.join(save_dir, preview_filename)
        plt.savefig(save_filename, bbox_inches="tight")

def wrapper_lowres_preview():
    # get all tif files in data-local/results_1000m/
    tif_files = glob.glob("/Volumes/OEMC/world-reforestation-monitor/results_1000m/*.tif")
    for tif in tif_files:
        generate_low_res_preview(tif)

    tif_files = glob.glob("/Volumes/OEMC/world-reforestation-monitor/results_100m/*.tif")
    for tif in tif_files:
        generate_low_res_preview(tif)

def zenodo_cleanup():
    # remove all files in zenodo-upload/depositions/test/*.json
    for f in glob.glob("zenodo-upload/depositions/test/*.json"):
        os.remove(f)

    for f in glob.glob("zenodo-upload/depositions/deploy/*.json"):
        os.remove(f)

    if FINAL_DEPOSITION:
        zen = Zenodo(url=Zenodo.url, token=ZENODO_ACCESS_TOKEN)
    else:
        zen = Zenodo(url=Zenodo.sandbox_url, token=ZENODO_ACCESS_TOKEN)
    current_deps = zen.depositions.list()

    # filter for string: 'Advancing Ecosystem Monitoring' in title
    current_deps = [dep for dep in current_deps if 'Advancing Ecosystem Monitoring' in dep.title]
    # delete all depositions
    for dep in current_deps:
        dep.discard()


if __name__ == "__main__":
    # test_tif_path = "data-local/results_1000m/lai_rtm.mlp_mean_1000m_s_20190101_20191231_go_epsg.4326_v01.tif"
    # wrapper_lowres_preview()
    # generate_low_res_preview(test_tif_path)
    zenodo_cleanup()
    main()
    # zenodo_cleanup()
