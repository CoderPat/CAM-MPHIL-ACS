import gzip
import codecs
from typing import Any

import xmltodict


def load_xml_gz(filename: str) -> Any:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        return xmltodict.parse(reader(f).read())
