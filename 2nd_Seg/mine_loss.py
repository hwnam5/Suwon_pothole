import torch
import torch.nn as nn
import torch.nn.functional as F

class MineNetwork(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_size=128):
        super(MineNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim1+input_dim2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x1, x2):

        N = min(x1.shape[0], x2.shape[0])
        
        idx1 = torch.randperm(x1.shape[0])[:N]
        idx2 = torch.randperm(x2.shape[0])[:N]
        
        x1 = x1[idx1]
        x2 = x2[idx2]

        joint = torch.cat([x1, x2], dim=1)
        return self.model(joint)
    
def mine_loss(mine_net, x1, x2):
    # shape: [batch, 1]
    joint_score = mine_net(x1, x2)

    x2_perm = x2[torch.randperm(x2.shape[0])]
    marginal_score = mine_net(x1, x2_perm)

    loss = -(torch.mean(joint_score) - torch.log(torch.mean(torch.exp(marginal_score))))
    return loss
    
def compute_mine_loss(mine_net, z, y, num_classes=3):
    mine_loss_total = 0.0
    valid_classes = []

    # reshape z: [B, C, H, W] → [B*H*W, C]
    z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))  # [B*H*W, C]
    y_flat = y.flatten()  # [B*H*W]

    for c in range(num_classes):
        mask = (y_flat == c)
        z_c = z_flat[mask] # [픽셀 수, class]
        valid_classes.append(z_c) #[class1, class2, class3]
    
    for i in range(len(valid_classes)):
        for j in range(i+1, len(valid_classes)):
            z1, z2 = valid_classes[i], valid_classes[j]
            mine_loss_total += mine_loss(mine_net, z1, z2)

    return mine_loss_total
