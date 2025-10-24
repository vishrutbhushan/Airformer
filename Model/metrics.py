import torch

def sudden_changes_mask(labels, threshold_start=75, threshold_change=20):
    labels = labels.squeeze(-1)
    b, t, n = labels.shape
    mask = torch.zeros(size=(b, t, n))
    mask_ones = torch.ones(size=(b, n))
    
    for t_idx in range(1, t):
        prev = labels[:, t_idx-1]
        curr = labels[:, t_idx]
        # High PM2.5 or large change
        mask[:, t_idx] = torch.where((curr > threshold_start) | (torch.abs(curr - prev) > threshold_change), 
                                     mask_ones, mask[:, t_idx])
    
    return mask.unsqueeze(-1)

def masked_mae(pred, real, mask=None):
    if mask is None:
        return torch.abs(pred - real).mean()
    
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(pred - real) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(pred, real, mask=None):
    if mask is None:
        return torch.sqrt(((pred - real) ** 2).mean())
    
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = ((pred - real) ** 2) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))
