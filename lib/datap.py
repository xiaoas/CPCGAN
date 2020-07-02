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
                pts.append(torch.from_numpy(dataf['pts'][()]))
                pts_num.append(torch.from_numpy(dataf['data_num'][()]))
                label.append(torch.from_numpy(dataf['label'][()]))
                label_seg.append(torch.from_numpy(dataf['label_seg'][()]))
            self.pts = torch.cat(pts)
            self.pts_num = torch.cat(pts_num)
            self.label = torch.cat(label)
            self.label_seg = torch.cat(label_seg)
        self.struct, self.struct_label_seg = self.genstruc(self.label_seg.max() + 1)
    def __len__(self):
        return len(self.pts)
    def __getitem__(self, idx):
        return {'pts': self.pts[idx], 'pts_num': self.pts_num[idx], 'label': self.label[idx], 'label_seg': self.label_seg[idx], \
            'struct': self.struct[idx], 'struct_label_seg': self.struct_label_seg[idx]}
    def genstruc(self, segcnt):
        resshape = self.pts.shape
        resshape[1] = 32
        result = torch.empty(resshape)
        reslbl = torch.empty(resshape[:-1])
        for idx, cld in enumerate(self.pts):
            partitio = torch.bincount(self.label_seg[idx, :self.pts_num[idx]]).to(torch.float)
            partitio = partitio / partitio.sum() * 32
            lft = partitio.frac()
            od = torch.argsort(lft, descending=True)
            partitio -= lft
            partitio[od[:32 - partitio.sum()]] += 1
            cstidx = 0
            #k means
            for segl, cnt in enumerate(partitio):
                subcld = cld[self.label_seg[idx] == segl]
                samples = subcld[torch.randperm(len(subcld))[:cnt]] # cnt samples in the subcld
                while(True):
                    diff = subcld[:, None, :] - samples[None, :, :]
                    diff = torch.norm(diff)
                    cate = torch.argmin(diff, dim = -1)
                    cnt = torch.bincount(cate)
                    nsamples = torch.empty_like(samples)
                    for sid in range(cnt):
                        nsamples[sid] = subcld[cate == sid].sum() / cnt[sid]
                    if torch.allclose(nsamples, samples):
                        break
                    samples = nsamples
                result[idx][cstidx:cstidx+cnt-1] = samples
                reslbl[idx][cstidx:cstidx+cnt-1] = segl

class Crop(object):
    def init(self):
        pass
    def __call__(self, sample):
        """\
        """
        pts, pts_num, label, label_seg, struct, struct_label_seg = sample['pts'], sample['pts_num'], sample['label'], sample['label_seg'], sample['struct'], sample['struct_label_seg']
        # tgt = 