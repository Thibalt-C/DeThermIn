import subprocess
import os
from DeThermIn.data import read_datacube
from DeThermIn.workflows.metadata import metadata

fusion_path = metadata["fusion_path"]
fusion_script = metadata["fusion_script"]

def launch(start, end, medium_path, coarse_path, ultra_coarse_path):

    process = subprocess.Popen(["python", fusion_script,
                                coarse_path, medium_path, ultra_coarse_path, start, end],
                               cwd=os.path.join(os.path.curdir, fusion_path),
                               stdout=subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=1,
                               env=os.environ.copy())

    lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        lines.append(line)

    process.wait()

    filepath = coarse_path.replace("LSR-HTR", "MSR-VHTR")

    dc = read_datacube(filepath)

    dc.compress(inplace=True)

    dc.save()

    return dc