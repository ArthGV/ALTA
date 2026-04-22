import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import pickle
import ast
import glob
import numpy as np
import logging
from tqdm import tqdm
from src.feat_compression.diffusion_model import load_diffusion_model
from src.inter import compress_sequence
from src.utils import format_output
import torch.nn.functional as F
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = argparse.ArgumentParser()
#DATA
parser.add_argument("--data_path", default="datasets")
parser.add_argument("--dataset", default="gtea")
parser.add_argument("--split", default=1, type=int)
#MODEL
parser.add_argument("--model_path", type=str)
parser.add_argument("--config_path", type=str)
#COMPRESSION HYPER PARAMETERS
parser.add_argument("--num_anchors",default=8, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--anchor_style", default='equidistant', type=str)
parser.add_argument("--inter_style", default='linear', type=str)
parser.add_argument("--print", default=False, type=bool)
parser.add_argument("--fixed", default=False, type=bool)
args = parser.parse_args()

ANCHOR_STRIDE = args.stride

log_filename = f"anchor_compressing_{args.dataset}_split{args.split}_num_anchors{args.num_anchors}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s] => %(message)s",
    handlers=[
        logging.FileHandler(filename=log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)

def process_video_seq(video_seq, actions_classes, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model, model_config = load_diffusion_model(device, model_config_path=args.config_path, model_weights_path=args.model_path)
    SAMPLE_NUM_TIMESTEP = model_config['model_parameters']['sample_num_timesteps']
    CLASS_SIZE = model_config['dataset']['class_size']

    seq = []
    with open(video_seq, "r") as f: 
        for line in f:
            seq.append(ast.literal_eval(line.strip()))
    
    video_name = video_seq.split("/")[-1].replace("txt", "npy")
    if args.print is True:
        logging.info(f"###VIDEO: {video_name}###")
    gt_feat_path = f"{args.data_path}/{args.dataset}/features/{video_name}" 
    feats = np.load(gt_feat_path, allow_pickle=True)

    num_anchors_per_action = args.num_anchors
    number_of_actions = len(seq)
    anchor_distibution = [2 for _ in range(number_of_actions)] #minimum for interpolation
    anchor_budget = number_of_actions * (num_anchors_per_action - 2)

    ###[Initialize compression]
    action_rec_error = []
    action_indexes = []
    actions_durations = []
    index = 0
    video_feats, video_anchors, video_anchors_ind = [], [], []
    for action_ind, s in enumerate(seq):
        action, duration = s
        if duration == 1:
            anchor_distibution[action_ind] = 1
            anchor_budget += 1
        num_anchors = anchor_distibution[action_ind]
        if args.fixed is True:
            num_anchors = min(duration, args.num_anchors)
        gt_T = torch.from_numpy(feats[:, index : index + duration].T).to(device)

        recon, anchors, anchors_ind = compress_sequence(device, model, gt_T, actions_classes, duration, args.anchor_style, args.inter_style, action, num_anchors, SAMPLE_NUM_TIMESTEP, CLASS_SIZE)

        action_indexes.append(index)
        actions_durations.append(s)
        action_rec_error.append(F.mse_loss(recon, gt_T).cpu())
        video_feats.append(recon.detach().cpu().numpy())
        video_anchors.append(anchors.detach().cpu().numpy())
        video_anchors_ind.append(anchors_ind)
        index += duration

    if args.fixed is False:
        ###[Dynamic Anchor Attribution]
        dynamic_iteration = 0
        while anchor_budget > 0:
            if args.print is True:
                logging.info("- - - - - - -")
                logging.info(f"Iteration: {dynamic_iteration+1}")
                logging.info(f"Budget: {anchor_budget}")
                logging.info(f"Anchors: {anchor_distibution}")
                dynamic_iteration += 1

            index_star = np.argmax(action_rec_error) #worst performing sub sequence
            index = action_indexes[index_star]
            action, duration = actions_durations[index_star]
            spent_budget = min(ANCHOR_STRIDE, anchor_budget, duration - anchor_distibution[index_star])
            anchor_distibution[index_star] += spent_budget
            num_anchors = anchor_distibution[index_star]
            anchor_budget -= spent_budget
            gt_T = torch.from_numpy(feats[:, index : index + duration].T).to(device)

            recon, anchors, anchors_ind = compress_sequence(device, model, gt_T, actions_classes, duration, args.anchor_style, args.inter_style, action, num_anchors, SAMPLE_NUM_TIMESTEP, CLASS_SIZE)

            action_rec_error[index_star] = F.mse_loss(recon, gt_T).cpu()
            video_feats[index_star] = recon.detach().cpu().numpy()
            video_anchors[index_star] = anchors.detach().cpu().numpy()
            video_anchors_ind[index_star] = anchors_ind

        if args.print is True:
            logging.info("- - - - - - -")
            logging.info(f"Iteration: {dynamic_iteration+1}")
            logging.info(f"Budget: {anchor_budget}")
            logging.info(f"Anchors: {anchor_distibution}")

    ###[Save Outputs]
    formated_video_feats = format_output(video_feats)
    formated_video_anchors = format_output(video_anchors)
    opt_feat_path = f"{args.data_path}/{args.dataset}/anchor_compressed_na{args.num_anchors}_st{args.stride}_sp{args.split}"
    os.makedirs(opt_feat_path, exist_ok=True)
    # store features
    video_feats_path = os.path.join(opt_feat_path, f"recon_{video_name}")
    np.save(video_feats_path, formated_video_feats)
    # store anchors
    video_anchor_path = os.path.join(opt_feat_path, f"anchors_{video_name}")
    np.save(video_anchor_path, formated_video_anchors)
    # store anchors ind
    video_anchor_ind_path = os.path.join(opt_feat_path, f"anchors_ind_{video_name}").replace('.npy', '.pkl')
    with open(video_anchor_ind_path, 'wb') as file:
        pickle.dump(video_anchors_ind, file)
    if args.print is True:
        logging.info(f"Path of recon: {video_feats_path}")
    return video_feats_path     

def worker(worker_args):
    video_seq, actions_classes, gpu_id = worker_args
    return process_video_seq(video_seq, actions_classes, gpu_id)

def main():
    video_sequences = glob.glob(f"{args.data_path}/{args.dataset}/seqs/*.txt")

    with open(f"datasets/{args.dataset}/mapping.txt", "r") as f:
        actions = f.read().splitlines()


    actions_classes = {a.split()[1]: int(a.split()[0]) for a in actions}
    
    logging.info(f"---------Condensing {args.dataset}, split{args.split} with num_anchors={args.num_anchors}---------")
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
