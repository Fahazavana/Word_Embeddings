import re
import unicodedata

def paragraph_normalizer(text: str) -> str:
	text = text.strip().strip('"').strip()
	text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

	text = re.sub(r"[^a-zA-Z0-9\s\n]", "   ", text)

	text = re.sub(r"\d+", "<NUM>", text)
	text = re.sub(r"<NUM>(\s*<NUM>)+", " <NUM> ", text)
	text = re.sub(r"\w*(<NUM>)\w*\1", " <NUM> ", text)

	text = re.sub(r"\s{2,}", " ", text)
	text = text.lower()

	return text

def normalize_data(in_file: str, out_file: str) -> None:
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