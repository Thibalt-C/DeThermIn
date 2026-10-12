import requests
import os
from shapely.geometry import mapping
import geojson
from shapely.geometry import Polygon
import time
import numpy as np


def authentificate(login, password, verbose=False):

    response = requests.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=(login, password))

    token_response = response.json()
    token = token_response['token']

    if verbose:
        print(token_response)

    return token


def check_tasks(token):

    response = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/task',
        headers={'Authorization': 'Bearer {0}'.format(token)})
    task_response = response.json()
    names = []
    for task in task_response:
        if task["status"].__contains__("done"):
            names.append(task['task_name'])

    return names, task_response


def make_request(gdf,
                 start, end,
                 token,
                 size,
                 products={"ECO_L2_LSTE.002": ["LST"
                                               #,"EmisWB"
                                               ]
                     #,
                           #"ECO_L2T_STARS.002":["albedo","NDVI"]
                           }
                 ):

    gdf_center = gdf.union_all().centroid.coords[0]
    lon, lat = gdf_center
    d_lon = size / (111320 * np.cos(np.radians(lat)))
    d_lat = size / 111320
    bounds = [lon - d_lon/2, lat - d_lat/2, lon + d_lon/2, lat + d_lat/2]
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    geometry = Polygon([(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)])

    feature_collection = geojson.FeatureCollection([geojson.Feature(geometry=mapping(geometry))])
    cords = list(geometry.centroid.coords)[0]
    cords_str = (f"{round(cords[0], 4)}_{round(cords[1], 4)}").replace('.','_')

    name = cords_str + '_' + start + '_' + end

    names, tasks = check_tasks(token)
    if name in names:
        n = names.index(name)
        return tasks[n]

    layers = [{"product": key, "layer": layer} for key in products.keys() for layer in products[key]]


    task = {
        'task_type': 'area',
        'task_name': name,
        'params': {
            "dates": [{
                "startDate": start,
                "endDate": end
            }],
            "layers": layers,
            "geo": feature_collection,
            "output": {
                "format": {
                    "type": "geotiff",
                    "filename_date": "calendar"
                },
                "projection": "geographic"
            }
        },
    }



    response = requests.post(
        'https://appeears.earthdatacloud.nasa.gov/api/task',
        json=task,
        headers={'Authorization': 'Bearer {0}'.format(token)})
    task_response = response.json()

    return task_response


def get_bundle(task_response, token):

    task_id = task_response["task_id"]
    response = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}'.format(task_id),
        headers={'Authorization': 'Bearer {0}'.format(token)}
    )
    bundle_response = response.json()

    return bundle_response


def look_for_status(task_response, token):

    task_id = task_response["task_id"]

    response = requests.get(
        f"https://appeears.earthdatacloud.nasa.gov/api/status/{task_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60
    )

    response.raise_for_status()

    status_response = response.json()

    if status_response.get("crashed"):
        raise RuntimeError("AppEEARS API crashed")

    if "status" in status_response:
        return status_response["status"]

    details = (
        status_response
        .get("progress", {})
        .get("details", [])
    )

    if len(details):
        return details[-1].get("desc", "UNKNOWN")

    return "UNKNOWN"

def download_task(task_response, bundle_response, token, path):

    task_id = task_response["task_id"]
    files = bundle_response["files"]
    files_ids = [file["file_id"] for file in files]
    filenames = [file["file_name"] for file in files]
    filetypes = [file["file_type"] for file in files]
    filepaths = []

    for i,files_id in enumerate(files_ids):

        if filetypes[i]=='tif':

            filepath = os.path.join(path, filenames[i])

            if not os.path.exists(filepath):

                response = requests.get(
                    'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(task_id,files_ids[i]),
                    headers={'Authorization': 'Bearer {0}'.format(token)},
                    allow_redirects=True,
                    stream=True
                )

                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                with open(filepath, 'wb') as f:
                    for data in response.iter_content(chunk_size=8192):
                        f.write(data)

            filepaths.append(filepath)

    return filepaths


def request_and_download(gdf, start, end, size, username, password, folder):

    token = authentificate(username, password)

    task_response = make_request(gdf, start, end, token, size)



    former_status = ""
    while not former_status.__contains__("done"):
        new_status = look_for_status(task_response, token)
        if new_status!=former_status:
            print(f"Status of AppEEARS request: {new_status}")
        former_status = new_status
        time.sleep(3)

    bundle_response = get_bundle(task_response, token)

    geometry = gdf.union_all()
    cords = list(geometry.centroid.coords)[0]
    cords_str = (f"{round(cords[0], 4)}_{round(cords[1], 4)}").replace('.', '_')
    path = folder + cords_str + '/'
    filepaths = download_task(task_response, bundle_response, token, path=path)

    return filepaths