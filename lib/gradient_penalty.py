import torch
from torch.autograd import Variable, grad

class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, lambdaGP, gamma=1, vertex_num=2500, device=torch.device('cpu')):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num
        self.device = device

    def __call__(self, netD, real_data1, real_data2, fake_data1, fake_data2):
        # print(real_data1.shape, real_data2.shape)
        # print(fake_data1.shape, fake_data2.shape)
        batch_size = real_data1.size(0)
        p_size = real_data1.shape[1]
        fake_data1 = fake_data1[:, :p_size, :]
        fake_data2 = fake_data2[:, :p_size, :]
        
        alpha = torch.rand(batch_size, 1, 1, requires_grad=True).to(self.device)
        # randomly mix real and fake data
        interpolates1 = real_data1 + alpha * (fake_data1 - real_data1)
        interpolates2 = real_data2 + alpha * (fake_data2 - real_data2)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates1, interpolates2)
        # compute gradients w.r.t the interpolated outputs
        
        gradients = grad(outputs=disc_interpolates, inputs=[interpolates1, interpolates2],
                         grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(batch_size,-1)
                         
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty