import json
import os
from pathlib import Path
from typing import Counter
from langid.langid import LanguageIdentifier, model
from torch.utils.data import Dataset
from lxml import etree
import langid
from tqdm import tqdm

from src.utils.paths import DATA_DIR


def parse_macocu_tmx(input_path, output_path):

    XML = "{http://www.w3.org/XML/1998/namespace}"
    data = list()
    context = etree.iterparse(input_path, events=("end",), tag="tu")
    counter = 0
    with open(output_path, "a") as out:
        for _, tu in tqdm(context):

            row = {}
            for tuv in tu.findall("tuv"):
                lang = tuv.get(f"{XML}lang")
                if lang == "hr_latin":
                    lang = "hr"
                seg = tuv.find("seg")
                row[lang] = seg.text
            data.append(row)
            tu.clear()
            counter += 1
            if counter % 100000 == 0:
                print(counter)
                out.write(
                    "\n".join(json.dumps(e, ensure_ascii=False) for e in data) + "\n"
                )
                out.flush()
                data = []


if __name__ == "__main__":
    from src.utils import paths
    import time

    dataset = parse_macocu_tmx(
        input_path=paths.DATA_DIR / "MaCoCu-hr-en.tmx",
        output_path=paths.DATA_DIR / "macocu",
    )
