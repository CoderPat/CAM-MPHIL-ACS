import gzip
import codecs
import pickle
from typing import Any

import xmltodict


def load_xml_gz(filename: str) -> Any:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        return xmltodict.parse(reader(f).read())

def save_pickle_gz(data: Any, filename: str) -> None:
    with gzip.GzipFile(filename, 'wb') as outfile:
        pickle.dump(data, outfile)
