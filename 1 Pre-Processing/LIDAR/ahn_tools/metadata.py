import json
import geopandas as gpd
from owslib.wfs import WebFeatureService
import io


def get_metadata(
        gdf,
        url="https://api.ellipsis-drive.com/v3/ogc/wfs/65945b69-81df-4270-97f0-f029033154c1?request=getCapabilities",
        layer="layerId_01170035-93d3-4a38-b04c-8e7be7a7ca78"
):

    bbox = gdf.to_crs("epsg:4326").union_all().bounds

    wfs = WebFeatureService(url, version="2.0.0")

    response = wfs.getfeature(typename=layer,
                              outputFormat="application/json",
                              bbox=bbox,
                              srsname="EPSG:4326")

    with io.BytesIO(response.read()) as f:
        metadata = gpd.read_file(f)
        metadata = metadata.loc[
            metadata.geometry.intersects(gdf.to_crs("epsg:4326").geometry.union_all())
        ].reset_index(drop=True)

    DEM_urls = metadata["Maaiveldmodel (DTM) ½m"].tolist()
    DSM_urls = metadata["Oppervlaktemodel (DSM) ½m"].tolist()
    lidar_urls = metadata["Puntenwolk"].tolist()

    DEM_filenames = [DEM_url.split("/")[-1] for DEM_url in DEM_urls]
    DSM_filenames = [DSM_url.split("/")[-1] for DSM_url in DSM_urls]
    lidar_filenames = [lidar_url.split("/")[-1] for lidar_url in lidar_urls]

    urls = (DEM_urls, DSM_urls, lidar_urls)
    filenames = (DEM_filenames, DSM_filenames, lidar_filenames)

    return urls, filenames