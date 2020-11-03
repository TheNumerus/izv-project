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

_data_header_types = [
    ["Kraj",                                      "U3"],
    ["ID nehody",                                 "U14"],
    ["Typ cesty",                                 "i1"],
    ["Číslo cesty",                               "i4"],
    ["Datum nehody",                              "datetime64[D]"],
    ["Den v týdnu",                               "i1"],
    ["Čas nehody",                                "2i1"],
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
        types = ["U8" for x in range(65)]

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
        # TODO create arrays
        types = list(map(lambda x: x[1], _data_header_types))
        for x in range(65):
            numpy_arrays.append(np.empty([len(parsed_data)], dtype=types[x]))

        rows_with_default = [2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
        rows_with_default_float = [46, 47, 48, 49, 50, 51]
        rows_with_strip_str = [4, 35, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64]

        # TODO clean up data
        for column, accident in enumerate(parsed_data):
            for row, data in enumerate(accident):
                if row in rows_with_default and data == "":
                    data = -1
                elif row in rows_with_strip_str:
                    data = data.strip('"')
                elif row == 6:
                    data = data.strip('"')
                    hour = int(data) // 100
                    if hour == 25:
                        hour = -1
                    minute = int(data) % 100
                    if minute == 60:
                        minute = -1
                    data = (hour, minute)
                elif row in rows_with_default_float:
                    data = data.strip('"')
                    if len(data) == 0:
                        data = float('nan')
                    else:
                        data = float(data.replace(',', '.'))

                numpy_arrays[row][column] = data

        header = list(map(lambda x: x[0], _data_header_types))

        return header, numpy_arrays

    def get_list(self, regions: list = None):
        header = list(map(lambda x: x[0], _data_header_types))
        data = (header, [[] for x in range(65)])

        for region in regions:
            print(f'getting data for region {region}')
            if region in self.cache.keys():
                data[1].extend(self.cache[region][1])
            else:
                cached_file = self.cache_filename.format(region)
                if os.path.isfile(cached_file):
                    # TODO read cache
                    pass
                else:
                    self.cache[region] = self.parse_region_data(region)
                    # TODO save cached data to disk
                    for x in range(65):
                        data[1][x].extend(self.cache[region][1][x])
        return data


if __name__ == "__main__":
    dd = DataDownloader()
    dd.get_list(['MSK', 'JHM', 'OLK'])
