import torch
import torch.nn.functional as F

def shift_mel(mel, direction="up", max_length=None):
    # mel: [1, C, T]
    B, C, T = mel.shape
    if max_length is None:
        max_length = T
    mel = mel.unsqueeze(1)  # B, 1, C, T
    
    if direction == "up":
        kernel = [0.2, 0.2, 0.6]
        pad_u, pad_d = len(kernel)-1 , 0
    elif direction == "down":
        kernel = [0.6, 0.2, 0.2]
        pad_u, pad_d = 0, len(kernel)-1 

    w = torch.tensor(kernel).view(1, 1, len(kernel), 1)
    w = w.to(mel.device)

    mel = F.pad(mel, (0, 0, pad_u, pad_d), mode="replicate")
    mel = F.conv2d(
        mel,
        w,
        bias=None,
        stride=1,
    )
    mel = mel.squeeze(1)

    return mel