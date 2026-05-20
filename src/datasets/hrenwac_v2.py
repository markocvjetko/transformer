import json
import os
from pathlib import Path
from langid.langid import LanguageIdentifier, model
from torch.utils.data import Dataset
from lxml import etree
import langid


class HrenWac(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_hrenwac_tmx(path):

    XML = "{http://www.w3.org/XML/1998/namespace}"
    folder = Path(path)
    tmx_files = list(folder.glob("*.tmx"))

    dataset = list()
    fail_counter = 0
    reject_pair_counter = 0
    lang_id_reject_counter = 0
    reject_len_counter = 0
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    identifier.set_languages(
        ["en", "hr"]
    )  # or ['en','hr','sr','bs','sl'] if you want to see confusions

    for i, file in enumerate(tmx_files):
        try:
            data = list()
            tree = etree.parse(file)
            for tu in tree.iterfind(".//tu"):
                row = dict()
                for tuv in tu.findall("tuv"):
                    lang = tuv.get(f"{XML}lang")
                    seg = tuv.findtext("seg")
                    row[lang] = seg
                a, b = row.values()
                if a == b:  # if text identical, reject
                    reject_pair_counter += 1
                    continue
                lang_a, prob_a = identifier.classify(a)
                lang_b, prob_b = identifier.classify(b)

                if len(a) / len(b) < 0.4 or len(a) / len(b) > 2.5:
                    reject_len_counter += 1
                    continue

                if (
                    lang_a == "en"
                    and lang_b == "hr"
                    and prob_a > 0.95
                    and prob_b > 0.95
                ):
                    data.append({"en": a, "hr": b})
                elif (
                    lang_a == "hr"
                    and lang_b == "en"
                    and prob_a > 0.95
                    and prob_b > 0.95
                ):
                    data.append({"en": b, "hr": a})
                else:
                    lang_id_reject_counter += 1
                    continue
            dataset.extend(data)

            print("rejected pairs: ", reject_pair_counter)
            print("reject len pairs: ", reject_len_counter)
            print("lang_id_reject_counter: ", lang_id_reject_counter)
        except Exception as e:
            fail_counter += 1
            print(f"Failed to read {file}: {e}, Failed :  {fail_counter}/{i}")
    return dataset


if __name__ == "__main__":
    from src.utils import paths
    import time

    dataset = parse_hrenwac_tmx(paths.DATA_DIR / "hrenwac_v2.0.tmx")

    start_time = time.time()
    with open(paths.DATA_DIR / "hrenwac", "w") as f:
        json.dump(dataset, f, ensure_ascii=False)
    end_time = time.time() - start_time
    print(end_time)
    start_time = time.time()
    dataset = HrenWac(paths.DATA_DIR / "hrenwac")
    print(len(dataset))
    end_time = time.time() - start_time
    # print(end_time)
    # from pprint import pprint
    # for row in data[:100]:
    #     pprint(row)
    # end_time = time.time() - start_time
    # print(end_time)
