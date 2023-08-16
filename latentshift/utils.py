import os
import sys
import requests
import pathlib


def download(url: str, filename: str):
    """Based on https://sumit-ghosh.com/articles/python-download-progress-bar/"""

    print("Downloading weights...")
    print("If this fails you can run `wget {} -O {}`".format(url, filename))

    pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)

    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
