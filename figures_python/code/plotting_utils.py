import urllib
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geemap import cartoee
from PIL import Image


def get_imgc(
    trait: str, resolution: str = "100m", version: str = "v01", year: int = 2020
):
    return ee.ImageCollection(
        f"projects/ee-speckerfelix/assets/open-earth/{trait}_predictions-mlp_{resolution}_{version}"
    ).filterDate(str(year))


def get_dimensions_bbox(bbox: ee.Geometry.BBox):
    """
    Get the length of the bounding box
    :param bbox: ee.Geometry.BBox
    :return: tuple(width, height)
    """
    coords = bbox.coordinates().get(0)
    bottomLeft = ee.List(coords).get(0)
    bottomRight = ee.List(coords).get(1)
    topRight = ee.List(coords).get(2)
    topLeft = ee.List(coords).get(3)

    bl = ee.Geometry.Point(bottomLeft)
    br = ee.Geometry.Point(bottomRight)
    tr = ee.Geometry.Point(topRight)
    tl = ee.Geometry.Point(topLeft)

    width = bl.distance(br).getInfo()
    height = bl.distance(tl).getInfo()
    return width, height


def blend_geom_to_img(
    img,
    geom,
    img_vis={"min": 0, "max": 2500, "bands": ["B4", "B3", "B2"]},
    geom_vis={"palette": "brown"},
    width=2,
):
    img = img.visualize(**img_vis)
    img = img.blend(ee.Image().paint(geom, 1, width).visualize(**geom_vis))
    return img


def plot_gee_image(
    trait,
    vis_params,
    band="b1",
    year=2022,
    resolution="1000m",
    label="Legend",
    output_path="output_image.png",
):
    fig = plt.figure(figsize=(15, 10))

    # predictedImage = ee.Image(
    #     f"projects/ee-speckerfelix/assets/open-earth/single_{trait}_2022_100m_v10"
    # )
    predictedImage = get_global_img_mosaic(
        trait, resolution=resolution, year=year, apply_scaling=True
    )
    print(predictedImage.getInfo())

    # predicted_imgc = ee.ImageCollection(
    #     f"projects/ee-speckerfelix/assets/open-earth/{trait}_predictions-mlp_1000m_v01"
    # ).filterDate("2022-01-01", "2022-12-31")
    # predictedImage = predicted_imgc.mosaic()  # .select(f"{trait}_mean").divide(scaling)

    # Image mask
    CGIAR_PET = ee.Image(
        "projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec"
    ).select("CGIAR_PET")

    # Robinson projection
    projection = ccrs.EqualEarth(central_longitude=0)
    region = [180, -88, -180, 88]

    ax = cartoee.get_map(
        ee.Image.constant(0).updateMask(CGIAR_PET.add(1).neq(0)),
        region=region,
        vis_params={"min": 1, "max": 15, "palette": ["#C1C1C1"]},
        proj=projection,
    )

    cartoee.add_colorbar(
        ax,
        vis_params=vis_params,
        loc="bottom",
        label=label,
        orientation="horizontal",
    )

    ax = cartoee.add_layer(
        ee_object=predictedImage.select(band),
        ax=ax,
        region=region,
        vis_params=vis_params,
        proj=projection,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def get_time_series(trait, geometry, resolution: str = "20m"):
    if resolution == "20m":
        suffix = "_20m_v01"
        scale = 20
    elif resolution == "100m":
        suffix = "_100m_v01"
        scale = 100
    elif resolution == "1000m":
        suffix = "_1000m_v01"
        scale = 1000
    else:
        raise ValueError("Invalid resolution")

    collections = {
        "lai": ee.ImageCollection(
            f"projects/ee-speckerfelix/assets/open-earth/lai_predictions-mlp{suffix}"
        ),
        "fapar": ee.ImageCollection(
            f"projects/ee-speckerfelix/assets/open-earth/fapar_predictions-mlp{suffix}"
        ),
        "fcover": ee.ImageCollection(
            f"projects/ee-speckerfelix/assets/open-earth/fcover_predictions-mlp{suffix}"
        ),
    }

    current_imgc = collections[trait].filterBounds(geometry)
    ts = (
        current_imgc.select([f"{trait.lower()}_mean", f"{trait.lower()}_stdDev"])
        .getRegion(geometry, scale)
        .getInfo()
    )

    dividers = {
        "lai": 1000,
        "fapar": 10000,
        "fcover": 10000,
    }

    ts_df = pd.DataFrame(ts[1:], columns=ts[0])
    ts_df["time"] = pd.to_datetime(ts_df["time"], unit="ms")
    ts_df["year"] = ts_df["time"].dt.year
    ts_df = (
        ts_df.set_index("year")[
            [f"{trait.lower()}_mean", f"{trait.lower()}_stdDev"]
        ].astype(float)
        / dividers[trait]
    )  # Scale values

    #
    ts_df = ts_df.groupby("year").mean()
    return ts_df.sort_index()


# Function to plot a single time series
def plot_single_trait(trait, geom):
    ts = get_time_series(trait, geom)
    print(ts)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the mean and uncertainty
    color = "blue"
    ax.plot(
        ts.index,
        ts[f"{trait.lower()}_mean"],
        label=f"{trait} Mean",
        marker="o",
    )
    ax.fill_between(
        ts.index,
        (ts[f"{trait.lower()}_mean"] - ts[f"{trait.lower()}_stdDev"]).clip(lower=0),
        (ts[f"{trait.lower()}_mean"] + ts[f"{trait.lower()}_stdDev"]).clip(lower=0),
        color=color,
        alpha=0.2,  # Add transparency to uncertainty
    )

    # set y axis limits
    y_axis_limits = {
        "lai": [0, 4.5],
        "fapar": [0, 1],
        "fcover": [0, 1],
    }

    ax.set_ylim(y_axis_limits[trait])

    # Set labels and legen

    labels = {"lai": "LAI", "fapar": "FAPAR", "fcover": "FCOVER"}
    ax.set_ylabel(labels[trait])

    plt.show()


def get_min_max_scale():
    return_dict = {
        "lai": {"min": 0, "max": 5.0, "scaling": 1000},
        "fapar": {"min": 0, "max": 1.0, "scaling": 10000},
        "fcover": {"min": 0, "max": 1.0, "scaling": 10000},
        "lai_std": {"min": 0, "max": 2.0},
        "fapar_std": {"min": 0, "max": 0.3},
        "fcover_std": {"min": 0, "max": 0.3},
    }
    return return_dict


def get_color_palettes():
    min_max_scale = get_min_max_scale()

    lai_hex = []

    timeseries_colors = {"lai": "#172313", "fapar": "#c53859", "fcover": "#74c476"}

    lai_hex = [
        "#fffdcd",
        "#e1cd73",
        "#aaac20",
        "#5f920c",
        "#187328",
        "#144b2a",
        "#172313",
    ]
    fcover_hex = ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
    fapar_hex = ["#ffffdd", "#e6ad12", "#c53859", "#3a26a1", "#000000"]
    std_hex = [
        "#440154",
        "#433982",
        "#30678D",
        "#218F8B",
        "#36B677",
        "#8ED542",
        "#FDE725",
    ]

    # Define color maps for panels 2-3 (LAI maps)
    cmap_lai = mcolors.LinearSegmentedColormap.from_list("custom_cmap", lai_hex, N=256)
    norm_lai = mcolors.Normalize(
        vmin=min_max_scale["lai"]["min"], vmax=min_max_scale["lai"]["max"]
    )  # Min/max for LAI

    cmap_fapar = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", fapar_hex, N=256
    )
    norm_fapar = mcolors.Normalize(
        vmin=min_max_scale["fapar"]["min"], vmax=min_max_scale["fapar"]["max"]
    )  # Min/max for FAPAR

    cmap_fcover = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", fcover_hex, N=256
    )
    norm_fcover = mcolors.Normalize(
        vmin=min_max_scale["fcover"]["min"], vmax=min_max_scale["fcover"]["max"]
    )  # Min/max for FCOVER

    cmap_std = mcolors.LinearSegmentedColormap.from_list("custom_cmap", std_hex, N=256)
    norm_lai_std = mcolors.Normalize(
        vmin=min_max_scale["lai_std"]["min"], vmax=min_max_scale["lai_std"]["max"]
    )  # Min/max for uncertainty
    norm_fapar_std = mcolors.Normalize(
        vmin=min_max_scale["fapar_std"]["min"], vmax=min_max_scale["fapar_std"]["max"]
    )  # Min/max for uncertainty
    norm_fcover_std = mcolors.Normalize(
        vmin=min_max_scale["fcover_std"]["min"], vmax=min_max_scale["fcover_std"]["max"]
    )  # Min/max for uncertainty

    cmap_diff = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["purple", "white", "green"], N=256
    )

    # cmap_std = mcolors.LinearSegmentedColormap.from_list("custom_cmap", std_hex, N=256)
    # norm_std = mcolors.Normalize(vmin=0, vmax=1)  # Min/max for uncertainty

    return {
        "lai_hex": lai_hex,
        "lai_norm": norm_lai,
        "lai_cmap": cmap_lai,
        "lai_ts": timeseries_colors["lai"],
        "lai_visparams": {
            "min": min_max_scale["lai"]["min"],
            "max": min_max_scale["lai"]["max"],
            "palette": lai_hex,
        },
        "fapar_hex": fapar_hex,
        "fapar_norm": norm_fapar,
        "fapar_cmap": cmap_fapar,
        "fapar_ts": timeseries_colors["fapar"],
        "fapar_visparams": {
            "min": min_max_scale["fapar"]["min"],
            "max": min_max_scale["fapar"]["max"],
            "palette": fapar_hex,
        },
        "fcover_hex": fcover_hex,
        "fcover_norm": norm_fcover,
        "fcover_cmap": cmap_fcover,
        "fcover_ts": timeseries_colors["fcover"],
        "fcover_visparams": {
            "min": min_max_scale["fcover"]["min"],
            "max": min_max_scale["fcover"]["max"],
            "palette": fcover_hex,
        },
        "std_hex": std_hex,
        "std_cmap": cmap_std,
        "std_lai_norm": norm_lai_std,
        "std_fapar_norm": norm_fapar_std,
        "std_fcover_norm": norm_fcover_std,
        "lai_std_visparams": {
            "min": min_max_scale["lai_std"]["min"],
            "max": min_max_scale["lai_std"]["max"],
            "palette": std_hex,
        },
        "fapar_std_visparams": {
            "min": min_max_scale["fapar_std"]["min"],
            "max": min_max_scale["fapar_std"]["max"],
            "palette": std_hex,
        },
        "fcover_std_visparams": {
            "min": min_max_scale["fcover_std"]["min"],
            "max": min_max_scale["fcover_std"]["max"],
            "palette": std_hex,
        },
        "diff_hex": ["purple", "white", "green"],
        "diff_cmap": cmap_diff,
        "diff_visparams_lai": {
            "min": -2,
            "max": 2,
            "palette": ["purple", "white", "green"],
        },
        "diff_visparams_fapar": {
            "min": -0.3,
            "max": 0.3,
            "palette": ["purple", "white", "green"],
        },
        "diff_visparams_fcover": {
            "min": -0.3,
            "max": 0.3,
            "palette": ["purple", "white", "green"],
        },
        "diff_lai_norm": mcolors.Normalize(vmin=-2, vmax=2),
        "diff_fapar_norm": mcolors.Normalize(vmin=-1, vmax=1),
        "diff_fcover_norm": mcolors.Normalize(vmin=-1, vmax=1),
    }


def url_to_image(url):
    return np.array(Image.open(urllib.request.urlopen(url)))


def get_global_img_mosaic(
    trait: str,
    resolution: str = "100m",
    version: str = "v01",
    year: int = 2020,
    apply_scaling: bool = True,
):
    img = (
        ee.ImageCollection(
            f"projects/ee-speckerfelix/assets/open-earth/{trait}_predictions-mlp_{resolution}_{version}"
        )
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )

    # only apply scale to mean and stdDev bands
    if apply_scaling:
        scaling = get_min_max_scale()[trait.lower()]["scaling"]
        return ee.Image(
            [
                img.select([f"{trait.lower()}_mean", f"{trait.lower()}_stdDev"])
                .divide(scaling)
                .rename([f"{trait.lower()}_mean", f"{trait.lower()}_stdDev"]),
                img.select([f"{trait.lower()}_count"]),
            ]
        )
    else:
        return img


def get_geometries():
    destas = ee.Geometry.Polygon(
        [
            [35.72917596072723, 7.363046051449763],
            [35.73006773829013, 7.362867636921568],
            [35.73155265121677, 7.36252430415105],
            [35.73239091243743, 7.362399464515272],
            [35.73341654359713, 7.362493087544376],
            [35.73505302703625, 7.362230006119446],
            [35.737773037850154, 7.3618643916273],
            [35.73957455238908, 7.361373836451502],
            [35.74027018737272, 7.361257890710257],
            [35.74065807413672, 7.36155665526766],
            [35.74133587173625, 7.362609021506792],
            [35.74137152759597, 7.363920017313549],
            [35.74034150002932, 7.365632287741479],
            [35.740096237541024, 7.367108278380328],
            [35.73966819270919, 7.368321109107219],
            [35.73969051974438, 7.371607468787599],
            [35.73782210849949, 7.371143780482284],
            [35.73298846291762, 7.370965382351534],
            [35.7324176573329, 7.371005546405908],
            [35.7310309101872, 7.3704347586212045],
            [35.73140992680587, 7.368784912615752],
            [35.72917596072723, 7.363046051449763],
        ]
    )

    # Define the site of interest (geometry A)
    eucal_bbox = (
        ee.Geometry.Polygon(
            [
                [
                    [145.59847461120214, -41.295669761682824],
                    [145.59847461120214, -41.30139262526389],
                    [145.61287270919408, -41.30139262526389],
                    [145.61287270919408, -41.295669761682824],
                ]
            ],
            None,
            False,
        )
        .centroid(maxError=1)
        .buffer(500)
        .bounds()
    )

    # Define two geometries for histogram extraction
    eucal_a = (
        ee.Geometry.Polygon(
            [
                [
                    [145.60051309005345, -41.29755594402977],
                    [145.60051309005345, -41.29957103584403],
                    [145.60525523559178, -41.29957103584403],
                    [145.60525523559178, -41.29755594402977],
                ]
            ],
            None,
            False,
        )
    )
    eucal_b = (
        ee.Geometry.Polygon(
            [
                [
                    [145.60617864028538, -41.297539823044225],
                    [145.60617864028538, -41.29955491535656],
                    [145.6109207858237, -41.29955491535656],
                    [145.6109207858237, -41.297539823044225],
                ]
            ],
            None,
            False,
        )
    )

    ee_fc = ee.FeatureCollection("projects/ee-speckerfelix/assets/figures/fire_geom")
    ee_geom_event = ee_fc.first().geometry()

    ee_geom_bbox = ee_fc.geometry().bounds()
    ee_geom = ee_fc.geometry()
    fire_bbox = ee_geom_bbox
    fire_geom = ee_geom

    point_geometries = {
        "campeche": ee.Geometry.Point([-90.1297730495951, 18.683391639805464]),
        # "congo_basin": ee.Geometry.Point([22.15659692843015, 1.6662957721269671]),
        "congo_basin": ee.Geometry.Point([22.148697900633977, 1.5865866533131594]),
        "wash": ee.Geometry.Point([-123.93749587447398, 47.33471630778331]),
        "peru": None,
    }

    campeche_geometry = ee.Geometry.Polygon(
        [
            [
                [-90.14449623835071, 18.67495312528892],
                [-90.14338043940052, 18.67426197415629],
                [-90.14342335474475, 18.673530164003317],
                [-90.14303711664661, 18.67226981689038],
                [-90.14308003199085, 18.670724862486605],
                [-90.14209297907337, 18.6699523800057],
                [-90.14226464045032, 18.66824477466564],
                [-90.14174965631946, 18.66726899246539],
                [-90.14076260340198, 18.672676381498913],
                [-90.13655689966663, 18.670724862486605],
                [-90.13471153986438, 18.671537998138163],
                [-90.12921837580188, 18.671700624800515],
                [-90.12389687311634, 18.67092814676507],
                [-90.12368229639515, 18.673692788754767],
                [-90.12153652918323, 18.67849014873624],
                [-90.1172449947594, 18.680156996266007],
                [-90.11514214289173, 18.682555601744617],
                [-90.11806038629993, 18.6841411019806],
                [-90.11694458734974, 18.686458344866995],
                [-90.11999157679065, 18.688369030070728],
                [-90.12300763631379, 18.68418175563771],
                [-90.12442384267365, 18.68296214168103],
                [-90.13107572103058, 18.688938166178186],
                [-90.13446603322541, 18.685807893923194],
                [-90.139615874534, 18.685685934093193],
                [-90.13918672109162, 18.690198389324944],
                [-90.14322076345002, 18.690645563026294],
            ]
        ]
    )

    mali = ee.Geometry.Polygon(
        [
            [0.08243754719765128, 16.177956604983912],
            [0.0834460577872509, 16.17488599377817],
            [0.0866217932608837, 16.176163703568815],
            [0.08947566365272941, 16.176802555364734],
            [0.09011939381630363, 16.177915389042205],
            [0.089261086931538, 16.18071805348356],
            [0.0884671530631298, 16.181439320985003],
            [0.08739426945717277, 16.181727827247865],
            [0.08649304722816886, 16.180759268840326],
            [0.08684709881813468, 16.180130733714847],
            [0.08627847050697746, 16.178997304661507],
            [0.08243754719765128, 16.177956604983912],
        ]
    )  # https://restor.eco/sites/ca6f920f-82fb-4951-b86d-b5208d6ebb13/?lat=16.178591302324584&lng=0.08616599109723122&zoom=16.3010692395308

    mangrove = ee.Geometry.Polygon(
        [
            [115.90169924943935, -8.770110876922349],
            [115.90170818311141, -8.770579122689197],
            [115.90189100748553, -8.770833236661504],
            [115.90202925367069, -8.771100845231299],
            [115.90224772220104, -8.771332733111844],
            [115.90269810429749, -8.771653737258863],
            [115.90306817324384, -8.771796441517202],
            [115.90374153316259, -8.772077349375387],
            [115.90382175510231, -8.772242344752396],
            [115.9039645121947, -8.772242357805188],
            [115.90400908993088, -8.772139774708299],
            [115.90400908993088, -8.771907900369131],
            [115.90400908993088, -8.771591324666586],
            [115.90390648893572, -8.771560137735772],
            [115.90374595484495, -8.771573503469309],
            [115.90363451037942, -8.771546733330734],
            [115.90348290973547, -8.771560137431852],
            [115.90343381998453, -8.771421867559354],
            [115.90327779753535, -8.771310426504378],
            [115.90307268517608, -8.771297050724414],
            [115.90312168475548, -8.771100826735799],
            [115.90285863902476, -8.77106514899934],
            [115.90298343914166, -8.770931385635896],
            [115.90279177212658, -8.770824322278493],
            [115.90291215057884, -8.770712885579753],
            [115.90282741642665, -8.770547914104196],
            [115.90291215057884, -8.77034276872102],
            [115.90315741924965, -8.770342744487538],
            [115.90343381998453, -8.770423065515386],
            [115.90352748755292, -8.770356127543936],
            [115.90350962033143, -8.770253578534396],
            [115.9038485559024, -8.770101992798201],
            [115.90401802352488, -8.770066280834252],
            [115.90396000027765, -8.769847814239135],
            [115.90388871197825, -8.769388528714382],
            [115.90363893206356, -8.769482197918242],
            [115.9033892421513, -8.769473236113475],
            [115.90332679707676, -8.769178933170569],
            [115.90360770955763, -8.769120990059555],
            [115.90375046676562, -8.769094234935453],
            [115.90393328972723, -8.768884610811034],
            [115.9039110008537, -8.768773156164254],
            [115.9031127511312, -8.768938188982766],
            [115.90271588133506, -8.769036234017026],
            [115.90250192513359, -8.769495545742801],
            [115.90220756587586, -8.769647174664193],
            [115.90214069881303, -8.769950405954065],
            [115.90193558557142, -8.770075196224585],
            [115.90169924943935, -8.770110876922349],
        ]
    )

    brazil = ee.Geometry.Polygon(
        [
            [-39.18193367896802, -14.67835734146883],
            [-39.18192923553053, -14.678696290554948],
            [-39.18186678933123, -14.679833326891591],
            [-39.18183108309047, -14.680644907828489],
            [-39.18160367353765, -14.681808705238959],
            [-39.1815457498294, -14.682169924897567],
            [-39.181568046495975, -14.683177643373407],
            [-39.18141196966496, -14.683971341608537],
            [-39.18137626316756, -14.684448538085935],
            [-39.181086484358936, -14.684390512098178],
            [-39.1806405461411, -14.684426169226555],
            [-39.18029720331157, -14.684207679441617],
            [-39.17995830219543, -14.684136394964717],
            [-39.17962384286409, -14.684020422461261],
            [-39.17927160727057, -14.683788542323041],
            [-39.17899959340039, -14.683601273619345],
            [-39.17890151576721, -14.683494236512031],
            [-39.17884351023463, -14.683177628779006],
            [-39.17884351023463, -14.682812004039246],
            [-39.178714168021166, -14.682406234303125],
            [-39.179262640670956, -14.6817641034129],
            [-39.17985125953177, -14.681064082804605],
            [-39.180404165871906, -14.680551229697866],
            [-39.1806940271537, -14.680314903541175],
            [-39.180841218728006, -14.679926990485571],
            [-39.1810195938266, -14.67971294231632],
            [-39.18115781831255, -14.6794498645968],
            [-39.18139411641877, -14.679097617729761],
            [-39.181701826330446, -14.678580310519454],
            [-39.18193367896802, -14.67835734146883],
        ]
    )

    return {
        "destas": destas,
        "eucal_bbox": eucal_bbox,
        "eucal_a": eucal_a,
        "eucal_b": eucal_b,
        "fire_bbox": fire_bbox,
        "fire_geom": fire_geom,
        "point_geometries": point_geometries,
        "campeche": campeche_geometry,
        "mali": mali,
        "mangrove": mangrove,
        "brazil": brazil,
    }


if __name__ == "__main__":
    ee.Initialize()

    geometries = get_geometries()
    mangrove = geometries["mangrove"]
    mali = geometries["mali"]
    destas = geometries["destas"]
    brazil = geometries["brazil"]

    destas_bbox = destas.centroid().buffer(1000).bounds()

    get_dimensions_bbox(destas_bbox)
