import torch
import pickle
import ast
import glob
import numpy as np
import logging
import sys
from tqdm import tqdm
from diffusion_model import load_diffusion_model
from inter import reconstruct_sequence_from_anchors
from utils import format_output
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = argparse.ArgumentParser()
#DATA
parser.add_argument("--data_path", default="datasets")
parser.add_argument("--dataset", default="gtea")
parser.add_argument("--compressed_folder", type=str)
parser.add_argument("--split", default=1, type=int)
#MODEL
parser.add_argument("--model_path", type=str)
parser.add_argument("--config_path", type=str)
#RECONSTRUCTION HYPER PARAMETERS
parser.add_argument("--inter_style", default='linear', type=str)
args = parser.parse_args()


log_filename = f"dataset_reconstruction_{args.dataset}_split{args.split}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s] => %(message)s",
    handlers=[
        logging.FileHandler(filename=log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)

def reconstuct_video(video_seq, actions_classes, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model, model_config = load_diffusion_model(device, model_config_path=args.config_path, model_weights_path=args.model_path)
    SAMPLE_NUM_TIMESTEP = model_config['model_parameters']['sample_num_timesteps']
    CLASS_SIZE = model_config['dataset']['class_size']

    seqs = []
    with open(video_seq, "r") as f: 
        for line in f:
            seqs.append(ast.literal_eval(line.strip()))

    video_recon = []
    anchor_index = 0

    video_name = video_seq.split("/")[-1].replace(".txt", "")
    anchor_ind_path = f'{args.data_path}/{args.dataset}/{args.compressed_folder}/anchors_ind_{video_name}.pkl'
    with open(anchor_ind_path, 'rb') as file:
            video_anchors_ind = pickle.load(file)
    video_anchors = np.load(f'{args.data_path}/{args.dataset}/{args.compressed_folder}/anchors_{video_name}.npy')
    video_anchors = torch.from_numpy(video_anchors.T)

    for i, seq in enumerate(seqs):
        action, duration = seq
        c = torch.linspace(0, 1, duration).unsqueeze(1).to(device)
        a = torch.zeros([duration, CLASS_SIZE]).to(device)
        a[:, actions_classes[action]] = 1

        anchors_ind = video_anchors_ind[i]
        anchors_ind_len = len(anchors_ind)
        anchors = video_anchors[anchor_index:anchor_index + anchors_ind_len]
        recon = reconstruct_sequence_from_anchors(device, model, a, c, duration, anchors,  anchors_ind, args.inter_style, SAMPLE_NUM_TIMESTEP)
        video_recon.append(recon)
        anchor_index += anchors_ind_len

    ###[Save Outputs]
    video_recon_folder_path = f"{args.data_path}/{args.dataset}/{args.compressed_folder}".replace('anchor_compressed', 'reconstructed')
    os.makedirs(video_recon_folder_path, exist_ok=True)
    formated_video_recon = format_output(video_recon)
    video_recon_path = os.path.join(video_recon_folder_path, f"recon_{video_name}.npy")
    np.save(video_recon_path, formated_video_recon)
    return video_recon_path


def worker(worker_args):
    video_seq, actions_classes, gpu_id = worker_args
    return reconstuct_video(video_seq, actions_classes, gpu_id)

def main():
    video_sequences = glob.glob(f"{args.data_path}/{args.dataset}/seqs/*.txt")

    with open(f"datasets/{args.dataset}/mapping.txt", "r") as f:
        actions = f.read().splitlines()


    actions_classes = {a.split()[1]: int(a.split()[0]) for a in actions}
    
    logging.info(f"---------Reconstructing {args.dataset} ,split{args.split}, from {args.compressed_folder}---------")
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    logging.info(f"---------Using {num_gpus} gpus---------")


    worker_args = [(video_seq, actions_classes, gpu_ids[i % len(gpu_ids)]) for i, video_seq in enumerate(video_sequences)]

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(worker, args): args for args in worker_args}
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing {futures[future]}: {e}")


if __name__ == "__main__":
    main()
