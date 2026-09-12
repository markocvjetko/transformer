import json

from lxml import etree
from tqdm import tqdm


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
                out.write("\n".join(json.dumps(e, ensure_ascii=False) for e in data) + "\n")
                out.flush()
                data = []


if __name__=="__main__":
    from src.utils import paths


    dataset = parse_macocu_tmx(
        input_path=paths.DATA_DIR / "MaCoCu-hr-en.tmx",
        output_path=paths.DATA_DIR / "macocu")
    