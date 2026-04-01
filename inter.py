import numpy as np
import torch
from typing import List, Literal

def interpolation(interpolation_style: Literal['linear', 'sigmoid'], x: float) -> float:
    if interpolation_style == 'linear':
        return x
    elif interpolation_style == 'sigmoid':
        return 1 / (1 + np.exp(-10 * (x - 0.5)))

def select_anchors(anchoring_style: Literal['border', 'equidistant'], duration, num_anchors) -> List[int]:
    if num_anchors > duration:
        raise ValueError("num_anchors cannot be greater than duration.")
    if anchoring_style == 'border':
        return [0, duration - 1]
    elif anchoring_style == 'equidistant':
        return np.linspace(0, duration - 1, num_anchors, dtype=int).tolist()
    
def compress_sequence(device, 
                      model, 
                      gt_T: torch.tensor, 
                      actions_classes: dict, 
                      duration: int, 
                      anchor_style: str,
                      inter_style: str,
                      action: int, 
                      num_anchors: int, 
                      sample_num_timestep: int, 
                      class_size: int) -> tuple[torch.tensor]:
    
    c = torch.linspace(0, 1, duration).unsqueeze(1).to(device)
    a = torch.zeros([duration, class_size]).to(device)
    a[:, actions_classes[action]] = 1
    
    anchors_ind = select_anchors(anchor_style, duration, num_anchors)
    anchors_x0 = [gt_T[i] for i in anchors_ind] # [Sub, D]
    with torch.no_grad():
        anchors = [model.ddim_invert(x0.unsqueeze(0), a[anchors_ind[i]].unsqueeze(0), c[anchors_ind[i]].unsqueeze(0), sample_num_timestep) for i, x0 in enumerate(anchors_x0)] # [Sub, D]
    anchors = torch.stack(anchors).squeeze(1).to(device)  # [Sub, D]
    
    outputs = reconstruct_sequence_from_anchors(device, model, a, c, duration, anchors, anchors_ind, inter_style, sample_num_timestep)
    return outputs, anchors, anchors_ind
    
def reconstruct_sequence_from_anchors(device, 
                                      model, 
                                      a: torch.tensor,
                                      c: torch.tensor,
                                      duration: int, 
                                      anchors: torch.tensor, 
                                      anchors_ind: list[int], 
                                      inter_style: str, 
                                      sample_num_timestep: int) -> torch.tensor:

    latents = [d for d in range(duration)]
    for it, ind in enumerate(anchors_ind):
        latents[ind] = anchors[it]
    list_prev_x0_ind = -1
    for it, f in enumerate(latents):
        if type(f) is not int:
            list_prev_x0_ind += 1
            continue
        prev_x0_ind = anchors_ind[list_prev_x0_ind]
        next_x0_ind = anchors_ind[list_prev_x0_ind + 1]
        relative_dist_to_prev = (it - prev_x0_ind) / (next_x0_ind - prev_x0_ind)
        next_coef = interpolation(inter_style, relative_dist_to_prev)
        prev_coef = 1 - next_coef
        hot_strat_i = prev_coef * anchors[list_prev_x0_ind] + next_coef * anchors[list_prev_x0_ind + 1]
        latents[it] = hot_strat_i
    latents = torch.stack(latents).squeeze(1).to(device)
    with torch.no_grad():
        outputs = model.ddim_sample(latents, a, c, sample_num_timestep)
    return outputs


      