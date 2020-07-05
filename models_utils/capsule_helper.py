import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if device == torch.device('cuda'):
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
# else:
#     torch.set_default_tensor_type(torch.FloatTensor)


class capsule_fusion(nn.Module):
    def __init__(self, opt, D, n_in, n_out, in_dim, out_dim, depth_encoding=False):
        super(capsule_fusion, self).__init__()

        self.D = D
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_in = n_in
        self.n_out = n_out
        self.epsilon = 1.0e-8
        self.pi = torch.tensor(math.pi)
        self.depth_encoding = depth_encoding
        R = torch.tensor([1/n_out for i in range(n_out)]).to(opt.device)
        self.R = R.view(1, self.n_out, 1, 1)

        if self.depth_encoding:
            self.l_enc = nn.Parameter(torch.randn(1, n_in, D))

        self.Wscore = nn.Parameter(torch.randn(n_in, D, 1))
        self.Bscore = nn.Parameter(torch.zeros(n_in, 1, 1))

        self.Wcap = nn.Parameter(torch.randn(n_in, D, in_dim))
        self.Bcap = nn.Parameter(torch.zeros(n_in, 1, in_dim))

        self.Wvote = nn.Parameter(torch.randn(n_in, n_out, in_dim, out_dim))
        self.Bvote = nn.Parameter(torch.zeros(n_in, n_out, 1, out_dim))

        self.beta_use = nn.Parameter(torch.zeros(n_in, 1, 1, 1))
        self.beta_ign = nn.Parameter(torch.zeros(1, n_out, 1, 1))

        nn.init.xavier_uniform_(self.Wscore.data)
        nn.init.xavier_uniform_(self.Wcap.data)
        nn.init.xavier_uniform_(self.Wvote.data)

    def forward(self, x, mask=None, iters=3):

        N, L, _ = x.size()

        if self.n_in != 1 and L != self.n_in:
            print("Invalid Input! (Guess what is invalid?)")

        # x shape = N x L x D

        if self.depth_encoding:
            x = x + self.l_enc

        x_t = torch.transpose(x, 0, 1)

        # x_t shape = L X N x D

        a_in = torch.matmul(x_t, self.Wscore)+self.Bscore

        # a_in shape L x N x 1

        mu_in = F.gelu(torch.matmul(x_t, self.Wcap)+self.Bcap)

        # mu_in shape L x N x in_dim

        f_a_in = torch.sigmoid(a_in)

        if mask is not None:
            mask = mask.view(N, L, 1)
            mask = torch.transpose(mask, 0, 1)
            f_a_in = f_a_in*mask

        # f_a_in shape L x N x 1
        f_a_in = f_a_in.view(L, 1, N, 1)

        mu_in = mu_in.view(L, 1, N, self.in_dim)
        mu_in = torch.repeat_interleave(mu_in, self.n_out, dim=1)

        # print(mu_in.size())

        # mu_in shape [L x n_out x N x in_dim]

        V = torch.matmul(mu_in, self.Wvote)+self.Bvote

        # V shape [L x n_out x N x out_dim]

        for t in range(iters):

            if t == 0:
                R = self.R
            else:
                log_p = -torch.sum((V-mu_out.unsqueeze(0)).pow(2) / (2*var_out.unsqueeze(0)+self.epsilon), dim=-1)\
                    - 1-0.5*self.pi-0.5*torch.sum(torch.log(var_out.unsqueeze(0)+self.epsilon), dim=-1)
                # log_p shape = L x n_out x N
                R = F.softmax(F.logsigmoid(a_out).unsqueeze(0)+log_p.unsqueeze(-1), dim=1)

                # R shape = L x n_out x N x 1

            D_use = f_a_in*R
            D_ign = f_a_in-D_use

            # D shape L x n_out x N x 1

            a_out = torch.sum(self.beta_use*D_use, dim=0) - \
                torch.sum(self.beta_ign*D_ign, dim=0)

            # a_out shape n_out x N x 1

            denom = (torch.sum(D_use, dim=0)+self.epsilon)

            mu_out = torch.sum(D_use*V, dim=0)/denom

            # mu_out shape n_out x N x out_dim

            var_out = torch.sum(D_use*(V-mu_out.unsqueeze(0)).pow(2), dim=0)/denom

            # var_out shape n_out x N x out_dim

        a_out = a_out.view(self.n_out, N)
        a_out = torch.transpose(a_out, 0, 1)

        mu_out = torch.transpose(mu_out, 0, 1)

        return a_out, mu_out
