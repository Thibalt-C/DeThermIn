from owslib.wms import WebMapService
import rioxarray as rxr
import xarray as xr
import numpy as np
from PIL import Image
import io


wms_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?&request=GetCapabilities&service=wms"
wms = WebMapService(wms_url, version="1.3.0")


def get_orthoimage(dataset, layer = "Actueel_orthoHR"):

    imgs = []

    try:

        xmin = dataset.x[0].values
        xmax = dataset.x[-1].values
        ymin = dataset.y[-1].values
        ymax = dataset.y[0].values

        W, H = len(dataset.x), len(dataset.y)

        img = wms.getmap(layers=[layer],
                             size=(W, H),
                             bbox=(xmin, ymin, xmax, ymax),
                             srs='EPSG:28992',
                             format='image/jpeg')

        imgs.append(img)

    except:

        subdataset_1 = dataset.where((dataset.x<dataset.x[W // 2].values) & (dataset.y<dataset.y[H // 2].values), drop=True)
        subdataset_2 = dataset.where((dataset.x >= dataset.x[W // 2].values) & (dataset.y < dataset.y[H // 2].values), drop=True)
        subdataset_3 = dataset.where((dataset.x < dataset.x[W // 2].values) & (dataset.y >= dataset.y[H // 2].values), drop=True)
        subdataset_4 = dataset.where((dataset.x >= dataset.x[W // 2].values) & (dataset.y >= dataset.y[H // 2].values), drop=True)

        for subdataset in [subdataset_1, subdataset_2, subdataset_3, subdataset_4]:

            xmin = subdataset.x[0].values
            xmax = subdataset.x[-1].values
            ymin = subdataset.y[-1].values
            ymax = subdataset.y[0].values

            W, H = len(subdataset.x), len(subdataset.y)

            img = wms.getmap(layers=[layer],
                             size=(W, H),
                             bbox=(xmin, ymin, xmax, ymax),
                             srs='EPSG:28992',
                             format='image/jpeg')

            imgs.append(img)


    orthophotos = []

    for img in imgs:

        orthophoto = Image.open(io.BytesIO(img.read()))
        orthophoto = np.array(orthophoto)

        if len(imgs)>1:
            orthophotos.append(orthophoto)

    if len(orthophotos)>0:
        orthophoto_1 = np.hstack(orthophotos[2:])
        orthophoto_2 = np.hstack(orthophotos[:2])
        orthophoto = np.concatenate((orthophoto_1, orthophoto_2))

    orthophoto_xarray = xr.DataArray(orthophoto, dims=['y', 'x', 'channel'],
                                     coords={"x":dataset.x, "y":dataset.y, "channel": ["red", "green", "blue"]}
                                     )
    orthophoto_xarray.assign_attrs({"standard_name": "Orthographic image",
                                    "original resolution": "8 cm",
                                    "units": "no units"})

    dataset["orthoimage"] = xr.DataArray(data=orthophoto_xarray.data,
                                         attrs=orthophoto_xarray.attrs,
                                         coords=orthophoto_xarray.coords,
                                         dims=('y', 'x', 'channel'))

    return dataset