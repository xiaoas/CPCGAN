import visdom 

vis = visdom.Visdom()

def plot(p, s):
    print('plot:', p.shape, s.shape)
    vis.scatter(p, s + 1, opts= {'markersize': 5, 'markerborderwidth': 0})