import torch
from torch import nn
from tqdm import tqdm

from nerfstudio.field_components.mlp import MLP


def identity_init(mlp, in_dim=3, out_dim=3, n_steps=5000):
    mlp = mlp.cuda()
    linspace = torch.linspace(0, 1, 100)[..., None].cuda()
    inp = torch.concatenate([linspace]*in_dim, dim=-1)
    out_gt = torch.concatenate([linspace]*out_dim, dim=-1)

    optimizer = torch.optim.Adam(mlp.parameters(), lr = 5e-2)
    loss_fnc = nn.MSELoss()
    for _ in tqdm(range(n_steps), desc="identity init"):
        out = mlp(inp)
        loss = loss_fnc(out_gt, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return mlp.cpu()

class MLP_Mapper(nn.Module):
    def __init__(self) -> None:
        super(MLP_Mapper, self).__init__()
        self.mlp = MLP(
                in_dim=1,
                num_layers=4,
                layer_width=16,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="torch",
            )
        identity_init(self.mlp, 1, 1)

    
    def forward(self, x, **kwargs):
        return self.mlp(x)


class RGB_MLP_Mapper(nn.Module):
    def __init__(self) -> None:
        super(RGB_MLP_Mapper, self).__init__()
        self.mlp = MLP(
                in_dim=3,
                num_layers=4,
                layer_width=16,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="torch",
            )
        identity_init(self.mlp, 3, 3)

    
    def forward(self, x, **kwargs):
        return self.mlp(x)

class GT_Mapper(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(GT_Mapper, self).__init__()
    
    def forward(self, x, **kwargs):
        return x ** (1/2.4)


class IdentityMapper(nn.Module):
    def __init__(self) -> None:
        super(IdentityMapper, self).__init__()
    
    def forward(self, x, **kwargs):
        return x



class Powpow(nn.Module):
    def __init__(self) -> None:
        super(Powpow, self).__init__()
        self.pow_coeff = nn.Parameter(torch.tensor([1.], dtype=torch.float32))
    
    def forward(self, x, **kwargs):
        return x**(self.pow_coeff)


MAPPERS_DICT = {"mlp" : MLP_Mapper,
                "rgb_mlp": RGB_MLP_Mapper,
                "gt" : GT_Mapper,
                "identity": IdentityMapper,
                "powpow": Powpow}