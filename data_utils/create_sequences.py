import os
import argparse
from tqdm import tqdm
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="gtea")
args = parser.parse_args()

def compute_sequence(file_path: str) -> list:
    with open(file_path, 'r') as file:
        prev_action = None
        current_duration = 0
        sequence = []
        for line in file:
            line = line.replace('\n', '')
            if line == '':
                continue
            if prev_action is None or prev_action == line:
                current_action = line
                current_duration += 1
            else:
                sequence.append((current_action, current_duration))
                current_duration = 1
            current_action = line
            prev_action = line
        sequence.append((current_action, current_duration))
    return sequence

file_paths = glob.glob(f"datasets/{args.dataset}/groundTruth/*.txt")
output_folder = f"datasets/{args.dataset}/seqs/"
os.makedirs(output_folder, exist_ok=True)

for file_path in tqdm(file_paths):
    sequence = compute_sequence(file_path)
    output_path = output_folder + file_path.split('/')[-1]
    with open(output_path, 'w') as file:
        for item in sequence:
            file.write(f"{item}\n")