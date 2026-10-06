import requests
import time
import tarfile
from tqdm import tqdm
import os

ground_truth_path = "."

s = requests.Session()

collection_id = "66eb550f-c02f-43f7-a822-3dfc0bbce685"

url = (
    "https://api.sedoo.fr/theia-catalogue-prod/"
    f"camcatt/data/prepare?collectionId={collection_id}"
)

payload = {
    "filter": {
        "type": "lst"
    }
}

headers = {
    "Content-Type": "application/json",
}

r = s.post(
    url,
    json=payload,
    headers=headers
)

requestID = r.text.strip().replace('"', '')

time.sleep(10)

download_url = f"https://api.sedoo.fr/theia-catalogue-prod/camcatt/data/collect/{requestID}"
response = s.get(download_url, stream=True)

if response.status_code == 200:
    print("Ground truth LST found in collection. Downloading...")

    size = response.headers["content-length"]
    filename = response.headers["content-disposition"].split("filename=")[1]
    filepath = os.path.join(ground_truth_path, filename)

    iterator = tqdm(response.iter_content(chunk_size=1024), total=int(size)//1024,
                    desc="Downloading ground truth LST (source: https://doi.org/10.6096/1012)"
                    )

    with open(filepath, "wb") as f:
        for chunk in iterator:
            f.write(chunk)

    with tarfile.open(filepath, "r") as tar:
        tar.extractall(path=ground_truth_path, filter="data")
    os.remove(filepath)