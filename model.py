import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from lib import datap, pltt
from lib.gradient_penalty import GradientPenalty
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
    def forward(self, z: torch.tensor):
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
        # print(z_.shape, GSS.shape, GSP.shape)
        x = torch.cat((z_, GSS, GSP), dim = -1) # (B, 32, 149)
        m = x[:, :, None, :] * x[:, None, :, :]
        m = torch.sum(m, -1) / 8 # / sqrt(149)
        m = torch.softmax(m, -1)
        m @= x
        x = self.mlp(m).reshape(-1,32,64,3)
        x += GSP[..., None, :]
        return torch.repeat_interleave(ss, 64, 1), x.reshape(-1, 2048, 3)

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
            nn.Linear(256, 1024)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, p):
        """\
            s: (B, 32, c)
            p: (B, 32, 3)
            output: 
        """
        x = torch.cat((s, p), -1)
        x = self.mlp1(x)
        x, idx = torch.max(x, 1)
        # x = torch.mean(x, 1)
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
    train_data = datap.Clddata('../data/', 'train_files.txt')
    val_data = datap.Clddata('../data/', 'val_files.txt')
    test_data = datap.Clddata('../data/', 'test_files.txt')
    # # print some of the struct cloud for fun
    # for idx in range(3):
    #     cnt = train_data.pts_num[idx]
    #     pltt.plot(train_data.pts[idx,:cnt], train_data.label_seg[idx,:cnt])
    #     pltt.plot(train_data.struct[idx,:cnt], train_data.struct_label_seg[idx,:cnt])
    BatchSize = 128
    dataloader = DataLoader(train_data, batch_size= BatchSize, shuffle=True)
    max_epoch_stage1 = 500
    max_epoch_stage2 = 1500
    lambda_s = 0.5
    sgen = StructGen().to(device)
    sdis = StructDis().to(device)
    fgen = FinalGen().to(device)
    fdis = StructDis().to(device)
    SGenDis = torch.distributions.MultivariateNormal(torch.zeros(96), torch.eye(96))
    gp = GradientPenalty(10, device=device)
    optimG = torch.optim.Adam(list(sgen.parameters()) + list(fgen.parameters()), lr = 1e-2, betas= (0, 0.99))
    optimD = torch.optim.Adam(list(sdis.parameters()) + list(fdis.parameters()), lr = 1e-2, betas= (0, 0.99))
    trans = datap.Crop()
    
    for idx in range(max_epoch_stage1):
        print('epoch', idx)
        for bid, datas in enumerate(dataloader):
            datas = trans(datas)
            pts = datas['pts'].to(device)
            label_seg = datas['label_seg'].to(device)
            sp, ss = datas['struct'].to(device), datas['struct_label_seg'].to(device)
            ss = torch.eye(c, device= device)[ss.to(torch.long)]
            label_seg =  torch.eye(c, device= device)[label_seg.to(torch.long)]
            Bsize = len(datas['pts'])
            for disUP in range(3): # train discriminator
                sgen_input = SGenDis.sample((Bsize,)).to(device)
                with torch.no_grad():
                    z_, gss, gsp = sgen(sgen_input)
                    fs, fp = fgen(z_, ss, sp)
                gploss_s = gp(sdis, ss, sp, gss, gsp)
                gploss_f = gp(fdis, label_seg, pts, fs, fp)
                Dsfake, Dsreal = sdis(gss, gsp).mean(), sdis(ss, sp).mean()
                Dffake, Dfreal = fdis(fs, fp).mean(), fdis(label_seg, pts).mean()
                lossD = lambda_s * (Dsfake - Dsreal) + Dffake - Dfreal + gploss_s + gploss_f
                optimD.zero_grad()
                lossD.backward()
                optimD.step()

            # train generator
            sgen_input = SGenDis.sample((Bsize,)).to(device)
            z_, gss, gsp = sgen(sgen_input)
            fs, fp = fgen(z_, ss, sp)
            Dsfake = sdis(gss, gsp).mean()
            Dffake = fdis(torch.repeat_interleave(ss, 64, 1), fp).mean()
            lossG = -lambda_s * Dsfake - Dffake
            optimG.zero_grad()
            lossG.backward()
            optimG.step()
            if bid % 32 == 0: 
                print('batch', bid, 'lossD', lossD.item(), 'lossG', lossG.item())
                print('\t', 'Dsfake', Dsfake.item(), 'Dsreal', Dsreal.item(), 'gploss_s', gploss_s.item())
                print('\t', 'Dffake', Dffake.item(), 'Dfreal', Dfreal.item(), 'gploss_f', gploss_f.item())
        """ # validation
        Vsize = len(val_datag)
        sgen_input = SGenDis.sample((Bsize,)).to(device)
        # z_, gss, gsp = sgen(sgen_input)
        sp, ss = val_datag['struct'].to(device), val_datag['struct_label_seg'].to(device)
        z_ = sgen_input[:, None, :].repeat(1, 32, 1) """
        torch.save({'sgenSD': sgen.state_dict(),
                    'sdisSD': sdis.state_dict(),
                    'fgenSD': fgen.state_dict(),
                    'fdisSD': fdis.state_dict()}, f'SDs_{idx % 10}.pt')
        if idx % 2 == 1:
            pltt.vis.text(f'epoch {idx} visualize')
            with torch.no_grad():
                # show generated structure cloud
                sgen_input = SGenDis.sample((2,)).to(device)
                z_, gss, gsp = sgen(sgen_input)
                print(gss.shape, gsp.shape)
                for p, s in zip(gsp, gss):
                    pltt.plot(p, s)
                # show ground truth and generated final cloud
                if True:
                    datas = train_data[:2]
                    for idx in range(2):
                        pltt.plot(datas['pts'][idx], datas['label_seg'][idx])
                    sp, ss = datas['struct'][:2].to(device), datas['struct_label_seg'][:2].to(device)
                    ss = torch.eye(c, device= device)[ss.to(torch.long)]
                    z_ = SGenDis.sample((2,)).to(device)
                    z_ = z_[:, None, :].repeat(1, 32, 1)
                    fs, fp = fgen(z_, ss, sp)
                    for p, s in zip(fp, fs):
                        pltt.plot(p, s)

    print('stage 2')
    pltt.vis.text('stage 2')


    for idx in range(max_epoch_stage2):
        print('epoch', idx)
        for bid, datas in enumerate(dataloader):
            datas = trans(datas)
            pts = datas['pts'].to(device)
            label_seg = datas['label_seg'].to(device)
            sp, ss = datas['struct'].to(device), datas['struct_label_seg'].to(device)
            ss = torch.eye(c, device= device)[ss.to(torch.long)]
            label_seg =  torch.eye(c, device= device)[label_seg.to(torch.long)]
            Bsize = len(datas['pts'])
            for disUP in range(3): # train discriminator
                sgen_input = SGenDis.sample((Bsize,)).to(device)
                with torch.no_grad():
                    z_, gss, gsp = sgen(sgen_input)
                    fs, fp = fgen(z_, gss, gsp)
                gploss_s = gp(sdis, ss, sp, gss, gsp)
                gploss_f = gp(fdis, label_seg, pts, fs, fp)
                Dsfake, Dsreal = sdis(gss, gsp).mean(), sdis(ss, sp).mean()
                # Dffake, Dfreal = fdis(torch.repeat_interleave(gss, 64, 1), fp), fdis(label_seg, pts)
                # print(fs.shape, fp.shape, label_seg.shape, pts.shape)
                Dffake, Dfreal = fdis(fs, fp).mean(), fdis(label_seg, pts).mean()
                lossD = lambda_s * (Dsfake - Dsreal) + Dffake - Dfreal +gploss_s + gploss_f
                optimD.zero_grad()
                lossD.backward()
                optimD.step()
            # train generator
            sgen_input = SGenDis.sample((Bsize,)).to(device)
            z_, gss, gsp = sgen(sgen_input)
            fs, fp = fgen(z_, gss, gsp)
            Dsfake = sdis(gss, gsp).mean()
            Dffake = fdis(torch.repeat_interleave(ss, 64, 1), fp).mean()
            lossG = -lambda_s * Dsfake - Dffake
            optimG.zero_grad()
            lossG.backward()
            optimG.step()
            if bid % 32 == 0: 
                print('batch', bid, 'lossD', lossD.item(), 'lossG', lossG.item())
        """ # validation
        Vsize = len(val_datag)
        sgen_input = SGenDis.sample((Bsize,)).to(device)
        # z_, gss, gsp = sgen(sgen_input)
        sp, ss = val_datag['struct'].to(device), val_datag['struct_label_seg'].to(device)
        z_ = sgen_input[:, None, :].repeat(1, 32, 1) """
        torch.save({'sgenSD': sgen.state_dict(),
                    'sdisSD': sdis.state_dict(),
                    'fgenSD': fgen.state_dict(),
                    'fdisSD': fdis.state_dict()}, f'SDs2_{idx % 10}.pt')
        if idx % 32 == 31:
            pltt.vis.text(f'epoch {idx} visualize')
            with torch.no_grad():
                # show generated structure cloud
                sgen_input = SGenDis.sample((3,)).to(device)
                z_, gss, gsp = sgen(sgen_input)
                print(gss.shape, gsp.shape)
                for p, s in zip(gsp, gss):
                    pltt.plot(p, s)
                # show generated final cloud
                z_ = SGenDis.sample((3,)).to(device)
                z_ = z_[:, None, :].repeat(1, 32, 1)
                fs, fp = fgen(z_, gss, gsp)
                for p, s in zip(fp, fs):
                    pltt.plot(p, s)