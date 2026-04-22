import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import torch
from src.utils import AverageMeter, BatchGenerater
from src.frame_compression.diffusion_model import init_diffusion_model
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
#DATA
parser.add_argument("--data_path", default="datasets", type=str)
parser.add_argument("--dataset", default="gtea")
parser.add_argument("--split", default=1, type=int)
#MODEL
parser.add_argument("--config_path", type=str)
#TRAINING HYPER PARAMETERS
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--epochs", default=10000, type=int)
parser.add_argument("--print_gap", default=1, type=int)
parser.add_argument("--save_gap", default=100, type=int)
args = parser.parse_args()

EPOCHS = args.epochs
PRINT_GAP = args.print_gap
SAVE_GAP = args.save_gap

# feat file
task_feat_path = f"{args.data_path}/{args.dataset}/splits/" + f"train.split{args.split}_vid.npy"
# batch generator
batch_gen = BatchGenerater(task_feat_path)
# model
model, model_config = init_diffusion_model(device, model_config_path=args.config_path)
# optimizer
optimizer = torch.optim.Adam(model.parameters())
#output paths
output_path = 'models/'
exp = f"diffusion_model_{args.dataset}_bz{args.batch_size}_ts{model_config['model_parameters']['num_timesteps']}_td{model_config['model_parameters']['timestep_dim']}_cd{model_config['model_parameters']['condition_dim']}"

# logging
log_filename = f"{args.dataset}_split{args.split}_{exp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s] => %(message)s",
    handlers=[
        logging.FileHandler(filename=log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)

logging.info(f"Working on {exp}")
logging.info(f"Training with dataset: {task_feat_path}")

for epoch in range(EPOCHS):
    epoch_loss = AverageMeter()
    batch_gen.reset()
    while batch_gen.has_next():
        batch_input = batch_gen.next_batch(args.batch_size)
        x, c, a = zip(*batch_input)
        x = torch.stack(list(x), dim=0).to(device).to(torch.float32)
        c = torch.stack(list(c), dim=0).to(device)
        a = torch.stack(list(a), dim=0).to(device)

        optimizer.zero_grad()
        loss = model.training_step(x, a, c)
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.item()/len(batch_input), len(batch_input))

    if (epoch + 1) % PRINT_GAP == 0:
        logging.info("[epoch %d]: epoch loss = %f" % (epoch + 1, epoch_loss.avg))

    if (epoch + 1) % SAVE_GAP == 0:
        model_name = output_path + f"{exp}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), model_name)
        logging.info(f"[epoch {epoch + 1}]: SAVED MODEL - - - - - - - - - - - -")

model_name = output_path + f"{exp}_epoch{epoch+1}.pth"
torch.save(model.state_dict(), model_name)
