import sys
sys.path.append("../../")
from utils import jdump, jload

def merge_jsonl_files(file1, file2, output_file):
    merged_data = []
    data1 = jload(file1)
    data2 = jload(file2)
    merged_data.extend(data1)
    merged_data.extend(data2)
    jdump(merged_data, output_file)

# Replace these filenames with the actual filenames you have
file1 = "data_0.json"
file2 = "data_1.json"
output_file = "data.json"

merge_jsonl_files(file1, file2, output_file)
print(f"Merged data saved to {output_file}")
