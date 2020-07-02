import torch 
import torch.nn as nn 

class StructGen(nn.Module):
    def __init__(self):
        super(CPCGAN, self).__init__()
        pass
    def forward(self, z):
        """\
            z: (B, 96)
            output: z_ (B, 32,96)
                    GSS (B, 32, c)
                    GSP (B, 32, 3)
        """
        pass

if __name__ == '__main__':
    device = torch.device('cuda:1')