import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from lib import datap
c = 50

class StructGen(nn.Module):
    def __init__(self):
        super(StructGen, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 256),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(256, 32 * c)
        self.fc2 = nn.Linear(256, 32 * 3)
        pass
    def forward(self, z: torch.tensor) -> Tuple[Tensor, ...]:
        """\
            z: (B, 96)
            output: z_ (B, 32,96)
                    GSS (B, 32, c)
                    GSP (B, 32, 3)
        """
        z_ = z[:, None, :].repeat(1, 32, 1)
        z = self.mlp1(z)
        gss = self.fc1(z).reshape(-1, 32, c)
        gsp = self.fc2(z).reshape(-1, 32, 3)
        return z_, gss, gsp
        pass

class FinalGen(nn.Module):
    def __init__(self):
        super(FinalGen, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(99 + c, 96),
            nn.ReLU(),
            nn.Linear(96, 192)
        )
        
    def forward(self, z_, GSS, GSP):
        """\
            output: (B, 2048,3)
        """
        x = torch.cat((z_, GSS, GSP), dim = -1) # (B, 32, 149)
        m = x[:, :, None, :] * x[:, None, :, :]
        m = torch.sum(m, -1) / 8 # / sqrt(149)
        m = torch.softmax(m, -1)
        m @= x
        x = self.mlp(m).reshape(-1,32,64,3)
        x += GSP[..., None, :]
        return x.reshape(-1, 2048, 3)

class StructDis(nn.Module):
    def __init__(self):
        super(StructDis, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(c+3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 2048)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, s, p):
        """\
            s: (B, 32, c)
            p: (B, 32, 3)
            output: 
        """
        x = torch.cat((s, p), -1)
        x = self.mlp1(x)
        x = torch.sum(x, dim = 1) / 32
        return self.mlp2(x)

# class FinalDis(nn.Module):
#     def __init__(self):
#         super(FinalDis, self).__init__()
#         pass
#     def forward(self, p):
#         """\
#             p: (B, 2048?, 3)
#             output: 
#         """
#         pass

if __name__ == '__main__':
    device = torch.device('cuda:1')
    train_data = datap.Clddata('../data', train_files.txt)
    val_data = datap.Clddata('../data', val_files.txt)
    test_data = datap.Clddata('../data', test_files.txt)
    dataloader = DataLoader(train_data, batch_size= 32, shuffle=True)
    max_epoch_stage1 = 10
    max_epoch_stage2 = 10
    sgen = StructGen().to(device)
    sdis = StructDis().to(device)
    fgen = FinalGen().to(device)
    fdis = StructDis().to(device)
    for idx in range(max_epoch_stage1):
        print('epoch', idx)
        for bid, datas in enumerate(dataloader):
            
            if bid % 10 == 9: # validate
                pass
