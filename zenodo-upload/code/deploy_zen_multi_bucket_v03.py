import glob
import os
from itertools import product
from typing import List, Union

from zen import LocalFiles, Zenodo

FINAL_DEPOSITION = True
# parse zenodo token from auth.zenodo_sandbox.txt
if not FINAL_DEPOSITION:
    with open("auth/zenodo_sandbox.txt", "r") as f:
        ZENODO_ACCESS_TOKEN = f.read().strip()
else:
    # raise ValueError("Final deposition not implemented yet.")
    with open("auth/zenodo.txt", "r") as f:
        ZENODO_ACCESS_TOKEN = f.read().strip()


AUTHOR_LIST = [
    {
        "name": "Felix Specker",
        "affiliation": "Institute of Integrative Biology, Department of Environmental Systems Science, ETH Zurich, Switzerland",
        "orcid": "0000-0002-9398-9975",
    },
    {
        "name": "Anna K. Schweiger",
        "affiliation": "Montana State University, Department of Land Resources and Environmental Sciences, Bozeman, MT, United States",
        "orcid": "0000-0002-5567-4200",
    },
    {
        "name": "Jean-Baptiste Féret",
        "affiliation": "TETIS, INRAE, AgroParisTech, CIRAD, CNRS, Université Montpellier, Montpellier, France",
        "orcid": "0000-0002-0151-1334",
    },
    {
        "name": "Luke A. Brown",
        "affiliation": "Department of Geography, King’s College London, London, WC2R 2LS, United Kingdom",
        "orcid": "0000-0003-4807-9056",
    },
    {
        "name": "Thomas Lauber",
        "affiliation": "Institute of Integrative Biology, Department of Environmental Systems Science, ETH Zurich, Switzerland",
        "orcid": "0000-0002-3118-432X",
    },
    {
        "name": "Charbel El Khoury",
        "affiliation": "Swiss Federal Research Institute WSL, Birmensdorf, 8903 Switzerland",
    },
    {
        "name": "Jadunandan Dash",
        "affiliation": "School of Science, Engineering & Environment, University of Salford, Manchester, M5 4WT, United Kingdom",
        "orcid": "0000-0002-5444-2109",
    },
    {
        "name": "Rémi Grousset",
        "affiliation": "ACRI-ST, F-06904, Sophia-Antipolis, France",
    },
    {
        "name": "Bert Gielen",
        "affiliation": "Plants and Ecosystems (PLECO), Department of Biology, University of Antwerp, B-2610, Wilrijk, Belgium",
        "orcid": "0000-0002-4890-3060",
    },
    {
        "name": "Thomas W. Crowther",
        "affiliation": "Biological and Environmental Science and Engineering Division, King Abdullah University of Science and Technology (KAUST), Thuwal, Kingdom of Saudi Arabia",
        "orcid": "0000-0001-5674-8913",
    },
    {
        "name": "Johan van den Hoogen",
        "affiliation": "Institute of Integrative Biology, Department of Environmental Systems Science, ETH Zurich, Switzerland",
        "orcid": "0000-0001-6624-8461",
    },
]

TRAIT_UPPER_MAPPING = {
    "laie": "LAIe",
    "fapar": "FAPAR",
    "fcover": "FCOVER",
}

NEW_TITLE = "Advancing Ecosystem Monitoring: Global Annual Maps of Biophysical Vegetation Properties (LAIe, FAPAR, FCOVER) for 2019-2025"

CORE_METADATA = {
    "title": NEW_TITLE,
    "upload_type": "dataset",
    "publication_type": "article",
    "license": "cc-by-4.0",
    "creators": AUTHOR_LIST,
    "grants": [
        {"id": "101059548"},
    ],
    "communities": [
        {"identifier": "oemc-project"},
    ],
}


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
    preview_files = grep_filenames(
        folder=folder_previews,
        suffix=".png",
        contains=os.path.basename(tif_path).replace(".tif", ""),
    )
    if len(preview_files) == 1:
        return preview_files[0]
    else:
        raise ValueError(
            f"Found {len(preview_files)} preview files for {tif_path}. Expected 1."
        )


def _clone_to_new_version(
    zen: Zenodo,
    old_dataset_path: str,
    new_dataset_path: str,
    new_filenames: List[str] | str,
    create_if_not_exists: bool = False,
    check_new_deposition_exists_and_delete: bool = True,
) -> LocalFiles:
    if isinstance(new_filenames, str):
        new_filenames = [new_filenames]
    if check_new_deposition_exists_and_delete and os.path.exists(new_dataset_path):
        # check if deposition for new_dataset_path already exists and delete if it does
        ds_existing = LocalFiles.from_file(new_dataset_path)
        try:
            ds_existing.set_deposition(api=zen, create_if_not_exists=False)
            ds_existing.deposition.discard()
            os.remove(new_dataset_path)
        except Exception as e:
            print(f"Failed to fully remove existing dataset/deposition: {e}")

    if not create_if_not_exists:
        ds_old = LocalFiles.from_file(old_dataset_path)
        ds_old.set_deposition(api=zen, create_if_not_exists=create_if_not_exists)
        new_dep = zen.depositions.retrieve(ds_old.deposition.id).new_version()
        new_dep.refresh()
        for remote_file in list(new_dep.files):
            new_dep.files.delete(remote_file)
        new_dep.refresh()

        ds_new = LocalFiles(files=list(new_filenames), dataset_path=new_dataset_path)
        ds_new.save()
        ds_new.set_deposition(api=zen, deposition=new_dep)
        ds_new.save()  # save to update local json with new deposition id and version
        ds_new.deposition.update()
        ds_new.upload()  # upload files to new deposition

    else:  # create new deposition from scratch with new metadata (for 2025 data, where no previous deposition exists)
        ds_new = LocalFiles(files=list(new_filenames), dataset_path=new_dataset_path)
        new_dep = ds_new.set_deposition(api=zen, create_if_not_exists=True)
        ds_new.save()  # save to update local json with new deposition id and version
        ds_new.upload()  # upload files to new deposition

    return ds_new


def _update_metadata(
    zen: Zenodo,
    dataset_path: str,
    updates: dict,
) -> LocalFiles:
    """
    Update multiple metadata fields in a Zenodo deposition.

    Parameters
    ----------
    zen
        Authenticated Zenodo client.
    dataset_path
        Path to local dataset JSON.
    updates
        Dict of metadata updates.
        Example:
            {
                "title": "...",
                "description": "...",
                "keywords": ["LAI", "FAPAR"],
            }
    """

    def _recursive_set(obj, updates_dict):
        for key, value in updates_dict.items():
            if isinstance(value, dict):
                # nested update
                sub_obj = getattr(obj, key)
                _recursive_set(sub_obj, value)
            else:
                setattr(obj, key, value)

    ds = LocalFiles.from_file(dataset_path)
    ds.set_deposition(api=zen, create_if_not_exists=False)

    _recursive_set(ds.deposition.metadata, updates)

    ds.deposition.update()
    ds.save()

    return ds


def update_v3():
    """
    Create base deposition with all 1000m resolution files.
    """
    if FINAL_DEPOSITION:
        zen = Zenodo(url=Zenodo.url, token=ZENODO_ACCESS_TOKEN)
    else:
        zen = Zenodo(url=Zenodo.sandbox_url, token=ZENODO_ACCESS_TOKEN)

    DEBUG = False
    # DATA_FOLDER_1000m = "/Volumes/OEMC/world-reforestation-monitor/results_1000m/"
    DATA_FOLDER_1000m = "data-local/merged-v03/results_1000m/"
    # DATA_FOLDER_100m = "/Volumes/OEMC/world-reforestation-monitor/results_100m/"
    DATA_FOLDER_100m = "data-local/merged-v03/results_100m/"
    doi_prefix = "https://doi.org/"

    ######
    # Update base deposition
    ######

    new_filenames_tif = grep_filenames(
        folder=DATA_FOLDER_1000m,
        suffix=".tif",
        contains=["_1000m"],
    )
    new_filenames_previews = list(
        map(
            lambda x: get_preview_file(x, "data-local/merged-v03/previews"),
            new_filenames_tif,
        )
    )
    new_filenames = [*new_filenames_tif, *new_filenames_previews]

    _clone_to_new_version(
        zen=zen,
        old_dataset_path="zenodo-upload/depositions/deploy-v1/deploy-deposition-base.json",
        new_dataset_path="zenodo-upload/depositions/deploy-v3/deploy-deposition-base.json",
        new_filenames=new_filenames[-1]
        if DEBUG
        else new_filenames,  # only upload the last file for testing
        check_new_deposition_exists_and_delete=True,
    )

    with open("zenodo-upload/templates/description-v3.html", "r") as f:
        description_v3 = f.read()

    # update title
    _update_metadata(
        zen=zen,
        dataset_path="zenodo-upload/depositions/deploy-v3/deploy-deposition-base.json",
        updates={**CORE_METADATA, "description": description_v3},
    )

    base_ds = LocalFiles.from_file(
        "zenodo-upload/depositions/deploy-v3/deploy-deposition-base.json"
    )
    base_ds.set_deposition(api=zen, create_if_not_exists=False)

    """
    ADD CODE AND DATA DEPOSITION HERE
    """

    # TODO

    """
    Create children depositions with 100m resolution files.
    """

    # now create children depositions: assert each is less than 50 GB
    traits = ["laie", "fapar", "fcover"]
    years = range(2019, 2026)
    children_ds = {}
    children_deps = {}
    for i, (trait, year) in enumerate(product(traits, years)):
        #######
        ## Deposit mean maps
        #######
        # if i > 2 and DEBUG:  # only create one deposition for testing
        #     break

        if (
            not ((year == 2025 and trait == "laie") or i < 1) and DEBUG
        ):  # only create new version for years that are already uploaded, for the last year (2025) create new deposition from scratch
            continue
        new_filenames_tif = grep_filenames(
            folder=DATA_FOLDER_100m,
            suffix=".tif",
            contains=[trait.lower(), "mean", str(year)],
        )
        new_filenames_previews = list(
            map(
                lambda x: get_preview_file(x, "data-local/merged-v03/previews"),
                new_filenames_tif,
            )
        )
        new_filenames = [*new_filenames_tif, *new_filenames_previews]

        # create new version of deposition with new files
        trait_old = trait if trait != "laie" else "lai"
        _clone_to_new_version(
            zen=zen,
            old_dataset_path=f"zenodo-upload/depositions/deploy-v1/deploy-deposition-base-{trait_old.lower()}-{year}-mean.json",
            new_dataset_path=f"zenodo-upload/depositions/deploy-v3/deploy-deposition-base-{trait.lower()}-{year}-mean.json",
            new_filenames=[f for f in new_filenames if "preview" in f]
            if DEBUG
            else new_filenames,  # only upload preview files for testing
            check_new_deposition_exists_and_delete=True,
            create_if_not_exists=False
            if year != 2025
            else True,  # only create new deposition for the last year (older years create new version)
        )

        _update_metadata(
            zen=zen,
            dataset_path=f"zenodo-upload/depositions/deploy-v3/deploy-deposition-base-{trait.lower()}-{year}-mean.json",
            updates={
                **CORE_METADATA,
                "description": f'<h3>Subdataset: {TRAIT_UPPER_MAPPING[trait.lower()]} {year} [mean] </h3>Mean {TRAIT_UPPER_MAPPING[trait.lower()]} predictions for {year} at 100m resolution. See base deposition for more information: <a href="{doi_prefix}{base_ds.deposition.doi}" target="_blank">{base_ds.deposition.doi}</a>',
                "related_identifiers": [
                    {
                        "relation": "isPartOf",
                        "identifier": base_ds.deposition.doi,
                        "resource_type": "dataset",
                    },
                ],
            },
        )

        ds_temp = LocalFiles.from_file(
            f"zenodo-upload/depositions/deploy-v3/deploy-deposition-base-{trait.lower()}-{year}-mean.json"
        )
        ds_temp.set_deposition(api=zen, create_if_not_exists=False)
        children_ds[f"{trait}-{year}-mean"] = ds_temp
        children_deps[f"{trait}-{year}-mean"] = ds_temp.deposition

        #######
        ## Deposit std maps
        #######

        new_filenames_tif = grep_filenames(
            folder=DATA_FOLDER_100m,
            suffix=".tif",
            contains=[trait.lower(), "std", str(year)],
        )
        new_filenames_previews = list(
            map(
                lambda x: get_preview_file(x, "data-local/merged-v03/previews"),
                new_filenames_tif,
            )
        )
        new_filenames = [*new_filenames_tif, *new_filenames_previews]

        # create new version of deposition with new files
        trait_old = trait if trait != "laie" else "lai"
        _clone_to_new_version(
            zen=zen,
            old_dataset_path=f"zenodo-upload/depositions/deploy-v1/deploy-deposition-base-{trait_old.lower()}-{year}-std-count.json",
            new_dataset_path=f"zenodo-upload/depositions/deploy-v3/deploy-deposition-base-{trait.lower()}-{year}-std.json",
            new_filenames=[f for f in new_filenames if "preview" in f]
            if DEBUG
            else new_filenames,  # only upload preview files for testing
            check_new_deposition_exists_and_delete=True,
            create_if_not_exists=False
            if year != 2025
            else True,  # only create new deposition for the last year (older years create new version)
        )

        _update_metadata(
            zen=zen,
            dataset_path=f"zenodo-upload/depositions/deploy-v3/deploy-deposition-base-{trait.lower()}-{year}-std.json",
            updates={
                **CORE_METADATA,
                "description": f'<h3>Subdataset: {TRAIT_UPPER_MAPPING[trait.lower()]} {year} [std] </h3>Standard deviation {TRAIT_UPPER_MAPPING[trait.lower()]} predictions for {year} at 100m resolution. See base deposition for more information: <a href="{doi_prefix}{base_ds.deposition.doi}" target="_blank">{base_ds.deposition.doi}</a>',
                "related_identifiers": [
                    {
                        "relation": "isPartOf",
                        "identifier": base_ds.deposition.doi,
                        "resource_type": "dataset",
                    }
                ],
            },
        )

        ds_temp = LocalFiles.from_file(
            f"zenodo-upload/depositions/deploy-v3/deploy-deposition-base-{trait.lower()}-{year}-std.json"
        )
        ds_temp.set_deposition(api=zen, create_if_not_exists=False)
        children_ds[f"{trait}-{year}-std"] = ds_temp
        children_deps[f"{trait}-{year}-std"] = ds_temp.deposition

    # link children year to year: continues, isContinuedBy
    for deposition_key, dep in children_deps.items():
        trait, year, var = deposition_key.split("-")
        previous_year = int(year) - 1
        previous_dep_key = f"{trait}-{previous_year}-{var}"
        next_year = int(year) + 1
        next_dep_key = f"{trait}-{next_year}-{var}"

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

    # for base deposition: update hasPart with all children depositions
    # update base deposition with related identifiers
    base_ds = LocalFiles.from_file(
        "zenodo-upload/depositions/deploy-v3/deploy-deposition-base.json"
    )
    base_ds.set_deposition(api=zen, create_if_not_exists=False)
    base_ds.deposition.metadata.related_identifiers.clear()
    for deposition_key, ds in children_ds.items():
        base_ds.deposition.metadata.related_identifiers.add(
            **{
                "relation": "hasPart",
                "identifier": ds.deposition.doi,
                "resource_type": "dataset",
            }
        )

    base_ds.deposition.update()

    #####
    # Updates Links in Base Depositions Metadata
    #####
    # update Base depositions description: add section with links to children / replace line: <p >ADD_RELATED_DOI_LINKS</p> with:
    base_dep_description = base_ds.deposition.metadata.description
    doi_prefix = "https://doi.org/"
    html_string = ""

    for trait in traits:
        html_string += f"<li><strong>{trait}</strong><ul>"

        for var in ["mean", "std"]:
            html_string += f"<li>{'Mean' if var == 'mean' else 'Std'}<ul>"

            for year in years:
                dep_key = f"{trait}-{year}-{var}"
                dep = children_deps.get(dep_key)  # Avoid KeyError with .get()

                if dep:  # Ensure dep exists before using its attributes
                    html_string += f'<li><a href="{doi_prefix}{dep.doi}" target="_blank">{year}</a></li>'

            html_string += "</ul></li>"  # Close variable ul and li

        html_string += "</ul></li>"  # Close trait ul and li

    html_string += "</ul>"  # Close the outer ul

    base_dep_description_updated = base_dep_description.replace(
        "ADD_RELATED_DOI_LINKS", html_string
    )
    base_ds.deposition.metadata.description = base_dep_description_updated
    base_ds.deposition.update()


def publish_all():
    """
    Publish all depositions (base and children).
    """
    if FINAL_DEPOSITION:
        zen = Zenodo(url=Zenodo.url, token=ZENODO_ACCESS_TOKEN)
    else:
        zen = Zenodo(url=Zenodo.sandbox_url, token=ZENODO_ACCESS_TOKEN)

    # publish base deposition
    base_ds = LocalFiles.from_file(
        "zenodo-upload/depositions/deploy-v3/deploy-deposition-base.json"
    )
    base_ds.set_deposition(api=zen, create_if_not_exists=False)
    base_ds.deposition.publish()

    # publish children depositions
    traits = ["laie", "fapar", "fcover"]
    years = range(2019, 2026)
    for trait, year in product(traits, years):
        for var in ["mean", "std"]:
            dep_path = f"zenodo-upload/depositions/deploy-v3/deploy-deposition-base-{trait.lower()}-{year}-{var}.json"
            if os.path.exists(dep_path):
                ds = LocalFiles.from_file(dep_path)
                ds.set_deposition(api=zen, create_if_not_exists=False)
                ds.deposition.publish()


if __name__ == "__main__":
    # update_v3()
    publish_all()
