import subprocess
import os
from typing import Optional
from ..data import read_datacube, DataCube
from ..metadata import metadata, extensions

def launch(
        enriched_path : str,
        mode : Optional[str] = "dynamic"
) -> DataCube :

    process = subprocess.Popen(["python",
                                metadata["super_resolution_script"],
                                enriched_path,
                                mode,
                                extensions["enriched_ancillary_data"],
                                extensions["end_product"],
                                extensions["training_data"]
                                ],
                               cwd=os.path.join(os.path.curdir, metadata["super_resolution_path"]),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        lines.append(line)

    process.wait()

    filepath = enriched_path.replace(
        extensions["enriched_ancillary_data"], extensions["end_product"]
    ).replace(extensions["training_data"], extensions["end_product"])

    dc = read_datacube(filepath)

    dc.compress(inplace=True)

    dc.save()

    return dc