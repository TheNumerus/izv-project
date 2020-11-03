import errno
from typing import *

import requests
import os
import re
import numpy as np
from zipfile import ZipFile
from bs4 import BeautifulSoup

_region_to_file = {
    "PHA": "00.csv",
    "STC": "01.csv",
    "JHC": "02.csv",
    "PLK": "03.csv",
    "ULK": "04.csv",
    "HKK": "05.csv",
    "JHM": "06.csv",
    "MSK": "07.csv",
    "OLK": "14.csv",
    "ZLK": "15.csv",
    "VYS": "16.csv",
    "PAK": "17.csv",
    "LBK": "18.csv",
    "KVK": "19.csv",
}

_header = [

]


class DataDownloader:
    def __init__(self, url="https://ehw.fit.vutbr.cz/izv/", folder="data", cache_filename="data_{}.pkl.gz"):
        self.cache_filename = cache_filename
        self.folder = folder
        self.url = url
        self.cache = dict()

    def download_data(self):
        res = requests.get(self.url, headers={'User-Agent': 'Mozilla 5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')

        try:
            os.mkdir(self.folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for link in soup.find_all('a', string="ZIP"):
            name = link['href'].rsplit('/', 1)[-1]

            filename = os.path.join(self.folder, name)

            if os.path.isfile(filename):
                print(name, "already in cache")
                continue

            print("downloading", name)

            file_url = self.url + link['href']
            file = requests.get(file_url, headers={'User-Agent': 'Mozilla 5.0'})

            f = open(filename, 'wb')
            f.write(file.content)

    def parse_region_data(self, region):
        parsed_data = []
        parsed_ids = set()
        types = "U16, U8"
        parsed_np = None

        print(f'parsing region {region}')

        # sort files from newest to oldest
        pat = re.compile(r"(\d{2})?-?(\d{4})")

        def sort_fn(x):
            date = pat.search(x)
            month, year = date.group(1), date.group(2)
            # this way, if no month is found in filename, file will have priority
            if month is None:
                return int(year) * 100 + 13
            else:
                return int(year) * 100 + int(month)

        sorted_files = sorted(os.listdir(self.folder), key=sort_fn, reverse=True)

        for file in sorted_files:
            fullpath = os.path.join(self.folder, file)
            with ZipFile(fullpath, 'r') as archive:
                with archive.open(_region_to_file[region]) as accidents:
                    # TODO fuck all remove duplicates
                    if parsed_np is None:
                        parsed_np = np.genfromtxt(accidents, delimiter=';', encoding='1250', autostrip=True, usecols=range(64))
                    else:
                        test = np.genfromtxt(accidents, delimiter=';', encoding='1250', autostrip=True, usecols=range(64))
                        parsed_np = np.append(parsed_np, test, axis=0)
                    #print(test)
                    # TODO clean up data
        arr = parsed_np.transpose()
        return (), list(arr)

    def get_list(self, regions: list = None):
        data = ([], [])

        for region in regions:
            print(f'getting data for region {region}')
            if region in self.cache.keys():
                data[0].extend(self.cache[region][0])
                data[1].extend(self.cache[region][1])
            else:
                cached_file = self.cache_filename.format(region)
                if os.path.isfile(cached_file):
                    # TODO read cache
                    pass
                else:
                    self.cache[region] = self.parse_region_data(region)
                    # TODO save cached data to disk
                    data[0].extend(self.cache[region][0])
                    data[1].extend(self.cache[region][1])
        return data


if __name__ == "__main__":
    dd = DataDownloader()
    dd.get_list(['MSK', 'JHM', 'OLK'])
