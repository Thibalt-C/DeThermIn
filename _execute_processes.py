from pathlib import Path
import subprocess
import os
import sys
import time
from DeThermIn.metadata import metadata, extensions, parameters

BASE_DIR = Path(__file__).resolve().parent

folder = BASE_DIR / "_processes"
path = str(BASE_DIR / "_figures") + "/"

CWDs = {
    #"_MAIN_": BASE_DIR,
   # "_FUSION_": BASE_DIR / metadata["fusion_path"],
    "_SR_": BASE_DIR / metadata["super_resolution_path"]
}

extra_args = {
    "_MAIN_": [],
    "_FUSION_": [parameters["processed_data_path"],
                 "4_4684_51_9249_06_01_2026_08_31_2026",
                 extensions["fusioned_LSTs"],
                 extensions["DTC-fitted"],
                 extensions["narrow-swath"]],
    "_SR_": [parameters["processed_data_path"],
             "1_4445_43_6085_05_01_2021_07_30_2021",
             extensions["training_data"]]
}


MAX_PROCESSES = 2

running = []

for key in CWDs.keys():
    for file in os.listdir(folder):
        if not (file.endswith(".py") and file.startswith(key)):
            continue

        while len(running) >= MAX_PROCESSES:
            running = [p for p in running if p.poll() is None]

            if len(running) >= MAX_PROCESSES:
                time.sleep(1)

        script = folder / file

        env = os.environ.copy()
        env["PYTHONPATH"] = str(CWDs[key])

        print(f"-> executing {script}...")

        args = [sys.executable, str(script), path]
        args.extend(extra_args[key])

        p = subprocess.Popen(
            args,
            env=env,
            cwd=str(CWDs[key])
        )

        running.append(p)

for p in running:
    p.wait()