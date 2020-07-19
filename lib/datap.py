from torch.utils import data
import h5py
import torch
class Clddata(data.Dataset):
    def __init__(self, path: str, fname: str):
        with open(path + fname, 'r') as f:
            pts = []
            pts_num = []
            label = []
            label_seg = []
            for datafname in f:
                datafname = datafname.strip()
                dataf = h5py.File(path + datafname, 'r')
                pts.append(torch.from_numpy(dataf['data'][()]))
                pts_num.append(torch.from_numpy(dataf['data_num'][()]))
                label.append(torch.from_numpy(dataf['label'][()]))
                label_seg.append(torch.from_numpy(dataf['label_seg'][()]))
            self.pts = torch.cat(pts)
            self.pts_num = torch.cat(pts_num)
            self.label = torch.cat(label)
            self.label_seg = torch.cat(label_seg)
            self.device = torch.device('cuda:1')
            # print('smallest pc:', self.pts_num.min().item())
        self.struct, self.struct_label_seg = self.genstruc(self.label_seg.max() + 1)
    def __len__(self):
        return len(self.pts)
    def __getitem__(self, idx):
        return {'pts': self.pts[idx], 'pts_num': self.pts_num[idx], 'label': self.label[idx], 'label_seg': self.label_seg[idx], \
            'struct': self.struct[idx], 'struct_label_seg': self.struct_label_seg[idx]}
    def genstruc(self, segcnt):
        resshape = list(self.pts.shape)
        resshape[1] = 32
        result = torch.empty(resshape)
        reslbl = torch.empty(resshape[:-1])
        for idx in range(len(self)):
            if idx % 256 == 0:
                print(idx / len(self))
            cld = self.pts[idx, :self.pts_num[idx]]
            lbls = self.label_seg[idx, :self.pts_num[idx]]
            partitio = torch.bincount(lbls).to(torch.float)
            partitio = partitio / partitio.sum() * 32
            lft = partitio.frac()
            od = torch.argsort(lft, descending=True)
            partitio = partitio.to(torch.int)
            partitio[od[:32-partitio.sum().item()]] += 1
            # print(partitio)
            cstidx = 0
            #k means
            for segl, cnt in enumerate(partitio):
                if cnt == 0:
                    continue
                subcld = cld[lbls == segl]
                sampleid = torch.randperm(len(subcld))[:cnt]
                samples = subcld[sampleid] # %cnt% samples in the subcld, initial means
                for iter in range(0):
                    diff = subcld[:, None, :] - samples[None, :, :]
                    diff = torch.norm(diff, dim= -1)
                    cate = torch.argmin(diff, dim = -1) # vis.scatter(subcld,cate + 1)
                    # num = torch.bincount(cate, minlength= cnt)
                    nsamples = torch.empty_like(samples)
                    for sid in range(cnt):
                        nsamples[sid] = subcld[cate == sid].mean(dim = 0)
                    if torch.allclose(nsamples, samples, atol= 1e-7, rtol= 1e-4) or torch.isnan(nsamples).any():
                        break
                        
                    samples = nsamples
                result[idx][cstidx:cstidx+cnt] = samples
                reslbl[idx][cstidx:cstidx+cnt] = segl
                cstidx += cnt

        return result, reslbl

class Crop(object):
    def init(self):
        pass
    def __call__(self, sample):
        """\
        """
        # pts, pts_num, label, label_seg, struct, struct_label_seg = sample['pts'], sample['pts_num'], sample['label'], sample['label_seg'], sample['struct'], sample['struct_label_seg']
        pts, pts_num, label_seg = sample['pts'], sample['pts_num'], sample['label_seg']
        minptsc = pts_num.min().item()
        minptsc = min(minptsc, 2048)
        shape = list(pts.shape)
        shape[1] = minptsc
        opts = torch.empty(shape)
        olbls = torch.empty(shape[:2])
        for idx in range(len(pts)):
            perm = torch.randperm(pts_num[idx])[:minptsc]
            opts[idx] = pts[idx][perm]
            olbls[idx] = label_seg[idx][perm]
        sample['pts'] = opts
        sample['label_seg'] = olbls
        return sample