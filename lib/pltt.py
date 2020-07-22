import visdom 

vis = visdom.Visdom()

def plot(p, s):
    # print('plot:', p.shape, s.shape)
    if len(p.shape) == len(s.shape):
        s = visdom.torch.argmax(s, -1)
    vis.scatter(p, s + 1, opts= {'markersize': 5, 'markerborderwidth': 0})