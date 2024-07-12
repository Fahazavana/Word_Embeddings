import re
import unicodedata

import json

# https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.json
with open("./data/en_stop_words.json") as stop_w:
    STOP_WORDS = json.load(stop_w)


def paragraph_normalizer(text: str) -> str:
    """
    Normalize a paragraph
    * remove extra \" and space
    * Remove stop words
    * Remove html tag
    * Words wit lenght less than two
    * Remove non alpha numeric
    * Replace all number, sequence of number,
            number followed by a word, or preceded (without sapce) by <num>
    *  Replace Diacritics by its simplified form
    * remove all extra space
    * Switch to lower case
    """
    text = text.strip().strip('"').strip()
    text = re.sub(r"\b(" + "|".join(STOP_WORDS) + r")\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"&lt;\w+&gt;|&lt;|\w+&gt;|&lt;|&gt;|&lt;/|\slt\s|\sgt\s", "  ", text)
    text = re.sub(r"\bgt\b|\blt\b", "  ", text)
    text = re.sub(r"\b\w{1,2}\b", "  ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\n]", "   ", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"\w*(<NUM>)+\w*", " <NUM> ", text)
    text = re.sub(r"<NUM>(\s*<NUM>)+", " <NUM> ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"\b\w{1,2}\b", "  ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower()

    return text.strip()


def normalize_data(in_file, out_file):
    """
    Normalize the contents of a file and write the results to another file.
    Args:
            in_file (str): Path to the input file.
            out_file (str): Path to the output file.
    """
    try:
        with open(in_file, "r") as file:
            with open(out_file, "w") as normalized_file:
                for line in file:
                    normalized_file.write(paragraph_normalizer(line.strip()) + "\n")
            print("Document Normalized Successfully!")
    except FileNotFoundError:
        print(f"File not found at {in_file}")
