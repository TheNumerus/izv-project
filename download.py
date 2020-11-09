import errno
import pickle
from typing import Tuple, List

import requests
import os
import re
import gzip
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

_data_header_types = [
    ["Kraj",                                      "U3"],
    ["ID nehody",                                 "U14"],
    ["Typ cesty",                                 "i1"],
    ["Číslo cesty",                               "i4"],
    ["Datum nehody",                              "datetime64[D]"],
    ["Den v týdnu",                               "i1"],
    ["Čas nehody",                                "i2"],
    ["Druh nehody",                               "i1"],
    ["Druh srážky",                               "i1"],
    ["Druh překážky",                             "i1"],
    ["Charakter nehody",                          "i1"],
    ["Zavinění nehody",                           "i1"],
    ["Alkohol přítomen",                          "i1"],
    ["Hlavní příčina nehody",                     "i2"],
    ["Usmrceno osob",                             "i1"],
    ["Těžce zraněno osob",                        "i1"],
    ["Lehce zraněno osob",                        "i1"],
    ["Celková hmotná škoda",                      "i4"],
    ["Typ povrchu",                               "i1"],
    ["Stav povrchu v době nehody",                "i1"],
    ["Stav komunikace",                           "i1"],
    ["Povětrnostní podmínky",                     "i1"],
    ["Viditelnost",                               "i1"],
    ["Rozhledové poměry",                         "i1"],
    ["Dělení komunikace",                         "i1"],
    ["Situování nehody na komunikaci",            "i1"],
    ["Řízení provozu v době nehody",              "i1"],
    ["Místní úprava přednosti v jízdě",           "i1"],
    ["Specifická místa a objekty v místě nehody", "i1"],
    ["Směrové poměry",                            "i1"],
    ["Počet zúčastněných vozidel",                "i1"],
    ["Místo dopravní nehody",                     "i1"],
    ["Druh křižující komunikace",                 "i1"],
    ["Druh vozidla",                              "i1"],
    ["Značka vozidla",                            "i1"],
    ["Rok výroby",                                "U2"],
    ["Charakteristika vozidla",                   "i1"],
    ["Smyk",                                      "i1"],
    ["Vozidlo po nehodě",                         "i1"],
    ["Únik provozních, přepravních hmot",         "i1"],
    ["Způsob vyprostění osob z vozidla",          "i1"],
    ["Směr jízdy nebo postavení vozidla",         "i1"],
    ["Škoda na vozidle (stovky kč)",              "i4"],
    ["Kategorie řidiče",                          "i1"],
    ["Vnější ovlivnění řidiče",                   "i1"],
    ["a",                                         "i1"],
    ["b",                                         "f4"],
    ["c",                                         "f4"],
    ["GPS souřadnice X",                          "f4"],
    ["GPS souřadnice Y",                          "f4"],
    ["f",                                         "f4"],
    ["g",                                         "f4"],
    ["Obec",                                      "U30"],
    ["Ulice",                                     "U30"],
    ["j",                                         "U30"],
    ["Typ komunikace",                            "U30"],
    ["Jméno komunikace",                          "U30"],
    ["n",                                         "U10"],
    ["o",                                         "U30"],
    ["Směr nehody",                               "U30"],
    ["Provoz v době nehody",                      "U15"],
    ["r",                                         "U15"],
    ["s",                                         "U15"],
    ["t",                                         "U30"],
    ["Lokalita nehody",                           "U30"],
]


class DataDownloader:
    def __init__(self, url="https://ehw.fit.vutbr.cz/izv/", folder="data", cache_filename="data_{}.pkl.gz"):
        self.cache_filename = cache_filename
        self.folder = folder
        self.url = url
        self.cache = dict()

    def download_data(self):
        """Downloads all zips with data"""
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
                continue

            file_url = self.url + link['href']
            file = requests.get(file_url, headers={'User-Agent': 'Mozilla 5.0'})

            f = open(filename, 'wb')
            f.write(file.content)

    def parse_region_data(self, region) -> Tuple[List[str], List[np.ndarray]]:
        """Returns parsed data for one region."""
        parsed_data = []
        parsed_ids = set()

        self.download_data()

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

        # ignore all pickles
        zips = list(filter(lambda x: "zip" in x, os.listdir(self.folder)))
        sorted_files = sorted(zips, key=sort_fn, reverse=True)

        for file in sorted_files:
            fullpath = os.path.join(self.folder, file)
            with ZipFile(fullpath, 'r') as archive:
                with archive.open(_region_to_file[region]) as region_stats:
                    region_stats = region_stats.read().decode('1250')
                    for accident in region_stats.splitlines(keepends=False):
                        # for some reason, some stats have 65 columns, so ignore the last
                        columns = accident.split(';')[:64]
                        if columns[0] not in parsed_ids:
                            parsed_ids.add(columns[0])
                            # add region code
                            columns.insert(0, region)
                            parsed_data.append(columns)

        numpy_arrays = []

        types = list(map(lambda x: x[1], _data_header_types))
        for x in range(65):
            numpy_arrays.append(np.empty([len(parsed_data)], dtype=types[x]))

        rows_with_default = [2, 3, 5] + [x for x in range(7, 35)] + [x for x in range(36, 46)]
        rows_with_default_float = [46, 47, 48, 49, 50, 51]
        rows_with_strip_str = [4, 6, 35, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64]

        for column, accident in enumerate(parsed_data):
            for row, data in enumerate(accident):
                if row in rows_with_default and data == "":
                    data = -1
                elif row in rows_with_strip_str:
                    data = data.strip('"')
                elif row in rows_with_default_float:
                    data = data.strip('"')
                    if len(data) == 0:
                        data = float('nan')
                    else:
                        data = float(data.replace(',', '.'))

                numpy_arrays[row][column] = data

        header = list(map(lambda x: x[0], _data_header_types))

        return header, numpy_arrays

    def get_list(self, regions: list = None) -> Tuple[List[str], List[np.ndarray]]:
        """Returns parsed data for selected regions. If `regions` is None, returns all regions"""
        header = list(map(lambda x: x[0], _data_header_types))
        data = (header, [])

        if regions is None:
            regions = list(_region_to_file.keys())

        for region in regions:
            # add data to program cache
            if region not in self.cache.keys():
                cached_file = os.path.join(self.folder, self.cache_filename.format(region))

                if os.path.isfile(cached_file):
                    cache = gzip.open(cached_file, 'rb')
                    self.cache[region] = pickle.load(cache)
                else:
                    self.cache[region] = self.parse_region_data(region)

                    cache = gzip.open(cached_file, 'wb')
                    pickle.dump(self.cache[region], cache)

            # add to output
            if len(data[1]) == 0:
                data = (header, self.cache[region][1])
            else:
                for x in range(65):
                    data[1][x] = np.append(data[1][x], self.cache[region][1][x])
        return data


if __name__ == "__main__":
    dd = DataDownloader()
    data = dd.get_list(['MSK', 'JHM', 'OLK'])

    print(f"Staženy data pro regiony 'MSK', 'JHM', 'OLK':\n")

    crashes_per_reg = {}
    for reg in data[1][0]:
        if reg not in crashes_per_reg:
            crashes_per_reg[reg] = 1
        else:
            crashes_per_reg[reg] += 1

    for key, value in crashes_per_reg.items():
        print(f"Nehod v regionu {key}: {value}")

    print("\nDatové sloupce:\n")
    for name in data[0]:
        print(name)
