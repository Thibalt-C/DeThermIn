import subprocess
import os
from DeThermIn.data import read_datacube
from DeThermIn.workflows.metadata import metadata

sr_path = metadata["sr_path"]
sr_script = metadata["sr_script"]

def launch(enriched_path, mode="dynamic"):

    process = subprocess.Popen(["python", sr_script, enriched_path, mode],
                               cwd=os.path.join(os.path.curdir, sr_path),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        lines.append(line)

    process.wait()

    filepath = enriched_path.replace("ENRICHED", "DeThermIn").replace("TRAININGREADY", "DeThermIn")

    dc = read_datacube(filepath)

    dc.compress(inplace=True)

    dc.save()

    return dc