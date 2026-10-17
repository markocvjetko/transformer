import json

from lxml import etree
from tqdm import tqdm


def parse_macocu_tmx(input_path, output_path):

    XML = "{http://www.w3.org/XML/1998/namespace}"
    data = list()
    context = etree.iterparse(input_path, events=("end",), tag="tu")
    with open(output_path, "w") as out:
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
            while tu.getprevious() is not None:
                del tu.getparent()[0]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            data = []


if __name__=="__main__":
    from src.utils import paths


    dataset = parse_macocu_tmx(
        input_path=paths.DATA_DIR / "MaCoCu-hr-en-v2/MaCoCu-hr-en.tmx",
        output_path=paths.DATA_DIR / "MaCoCu-hr-en-v2/parsed_macocu.jsonl")
    