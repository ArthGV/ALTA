import torch
import ast
import glob
import numpy as np
import logging
import sys
from tqdm import tqdm
from diffusion_model import load_diffusion_model
from inter import interpolation, select_anchors
from utils import format_output
import torch.nn.functional as F
import os
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

log_filename = f"{args.dataset}_split{args.split}_num_anchors{args.num_anchors}.log"
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
    best_video_feats, best_video_noises = [], []
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
    best_video_feats, best_video_noises = [], []
    for action_ind, s in enumerate(seq):
        action, duration = s
        if duration == 1:
            anchor_distibution[action_ind] = 1
            anchor_budget += 1
        num_anchors = anchor_distibution[action_ind]
        if args.fixed is True:
            num_anchors = min(duration, args.num_anchors)
        
        outputs, noise, gt_T = compress_action(device, model, actions_classes, feats, duration, action, num_anchors, SAMPLE_NUM_TIMESTEP, CLASS_SIZE, index)

        action_indexes.append(index)
        actions_durations.append(s)
        action_rec_error.append(F.mse_loss(outputs, gt_T).cpu())
        best_video_feats.append(outputs.detach().cpu().numpy())
        best_video_noises.append(noise.detach().cpu().numpy())
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

            outputs, noise, gt_T = compress_action(device, model, actions_classes, feats, duration, action, num_anchors, SAMPLE_NUM_TIMESTEP, CLASS_SIZE, index)
            
            action_rec_error[index_star] = F.mse_loss(outputs, gt_T).cpu()
            best_video_feats[index_star] = outputs.detach().cpu().numpy()
            best_video_noises[index_star] = noise.detach().cpu().numpy()
        if args.print is True:
            logging.info("- - - - - - -")
            logging.info(f"Iteration: {dynamic_iteration+1}")
            logging.info(f"Budget: {anchor_budget}")
            logging.info(f"Anchors: {anchor_distibution}")

    ###[Save Outputs]
    best_feats = format_output(best_video_feats)
    best_noises = format_output(best_video_noises)
    opt_feat_path = f"{args.data_path}/{args.dataset}/anchor_compressed_na{args.num_anchors}_st{args.stride}__sp{args.split}"
    os.makedirs(opt_feat_path, exist_ok=True)
    # store features
    best_feat_name = os.path.join(opt_feat_path, video_name)
    np.save(best_feat_name, best_feats)
    # store noises
    best_noise_name = os.path.join(opt_feat_path, f"noise_{video_name}")
    np.save(best_noise_name, best_noises)
    return best_feat_name    

def compress_action(device, model, actions_classes, feats, duration, action, num_anchors, sample_num_timestep, class_size, index):
    gt_T = torch.from_numpy(feats[:, index : index + duration].T).to(device)
    c = torch.linspace(0, 1, duration).unsqueeze(1).to(device)
    a = torch.zeros([duration, class_size]).to(device)
    a[:, actions_classes[action]] = 1
    
    num_anchors_x0_ind = select_anchors(args.anchor_style, duration, num_anchors)
    num_anchors_x0 = [gt_T[i] for i in num_anchors_x0_ind] # [Sub, D]
    with torch.no_grad():
        hot_start = [model.ddim_invert(x0.unsqueeze(0), a[num_anchors_x0_ind[i]].unsqueeze(0), c[num_anchors_x0_ind[i]].unsqueeze(0), sample_num_timestep) for i, x0 in enumerate(num_anchors_x0)] # [Sub, D]
    noise = torch.stack(hot_start).squeeze(1).to(device)  # [Sub, D]
    
    noises = [d for d in range(duration)]
    for it, ind in enumerate(num_anchors_x0_ind):
        noises[ind] = noise[it]
    list_prev_x0_ind = -1
    for it, f in enumerate(noises):
        if type(f) is not int:
            list_prev_x0_ind += 1
            continue
        prev_x0_ind = num_anchors_x0_ind[list_prev_x0_ind]
        next_x0_ind = num_anchors_x0_ind[list_prev_x0_ind + 1]
        relative_dist_to_prev = (it - prev_x0_ind) / (next_x0_ind - prev_x0_ind)
        next_coef = interpolation(args.inter_style, relative_dist_to_prev)
        prev_coef = 1 - next_coef
        hot_strat_i = prev_coef * noise[list_prev_x0_ind] + next_coef * noise[list_prev_x0_ind + 1]
        noises[it] = hot_strat_i
    noises = torch.stack(noises).squeeze(1).to(device)
    with torch.no_grad():
        outputs = model.ddim_sample(noises, a, c, sample_num_timestep)
    return outputs, noise, gt_T

def worker(worker_args):
    video_seq, actions_classes, gpu_id = worker_args
    return process_video_seq(video_seq, actions_classes, gpu_id)

def main():
    video_sequences = glob.glob(f"{args.data_path}/{args.dataset}/seqs/*.txt")

    with open(f"datasets/{args.dataset}/mapping.txt", "r") as f:
        actions = f.read().splitlines()


    actions_classes = {a.split()[1]: int(a.split()[0]) for a in actions}
    
    logging.info(f"---------Processing {args.dataset} with diffusion with num_anchors {args.num_anchors} base on sp {args.split}---------")
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
