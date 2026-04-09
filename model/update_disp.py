import torch
import torch.nn as nn
import torch.nn.functional as F

class DispHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,3), padding=(0,1))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,3), padding=(0,1))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,3), padding=(0,1))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3,1), padding=(1,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3,1), padding=(1,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3,1), padding=(1,0))
        
    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q
        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q
        return h

class BasicDispEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(BasicDispEncoder, self).__init__()

        cor_planes = corr_levels * (corr_radius*2+1) * 5
        self.convc1 = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 128, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(128+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):

        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        dis = F.relu(self.convd1(disp))
        dis = F.relu(self.convd2(dis))

        cor_dis = torch.cat([cor, dis], dim=1)
        out = F.relu(self.conv(cor_dis))
        return torch.cat([out, disp], dim=1)

class DispUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, hidden_dim=128):
        super(DispUpdateBlock, self).__init__()
        self.encoder = BasicDispEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.disp_head = DispHead(hidden_dim, hidden_dim=128)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, corr, inp, disp):
        # motion_features = self.encoder(disp, corr)
        # inp = torch.cat([inp, motion_features], dim=1)
        # inp = self.encoder(disp, corr)
        disp_inp = self.encoder(disp, corr)
        inp = torch.cat([inp, disp_inp], dim=1)

        net = self.gru(net, inp)
        delta_disp = self.disp_head(net)

        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_disp

class SubPixCor:
    def __init__(self, fmap1:torch.Tensor, fmap2:torch.Tensor, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.fmap2 = fmap2
        self.fmap1 = fmap1.permute(0, 2, 3, 1).reshape(-1, fmap1.shape[1]).unsqueeze(-1)        
    
    def __call__(self, coords1):
        batch,_, h1, w1 = coords1.shape
        out_pyramid = []
        for i in range(self.num_levels):
            f2 = self.sample_horizontal_from_coords(self.fmap2, coords1, lvl=i)

            corr = SubPixCor.compute_corr(self.fmap1, f2)        
            if i > 0:
                corr = F.avg_pool1d(corr, i+1, stride=i+1)
            corr = corr.view(batch, h1, w1, -1)

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    def sample_horizontal_from_coords(self, img, coords, lvl):
        r = self.radius
        N, C, H, W = img.shape
        dilation = lvl+1
        v_sz = (2*r+1)*dilation
        dx = torch.linspace(-r*(dilation), r*(dilation), steps=v_sz, device=img.device)  # (2r+1,)

        coords = coords.expand(-1, v_sz, -1, -1)  # (N, 2r+1, H, W)
        coords = coords + dx.view(1, -1, 1, 1)     # (N, 2r+1, H, W)
        coords_x = (coords / (W - 1)) * 2 - 1      # (N, 2r+1, H, W)
        grid_y = torch.linspace(-1, 1, steps=H, device=img.device).view(1, 1, H, 1).expand(N, v_sz, H, W)
        grid = torch.stack([coords_x, grid_y], dim=-1)
        grid = grid.permute(0, 2, 3, 1, 4).reshape(N, H, W * (v_sz), 2)
        sampled = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (N, C, H, W * 2r+1)
        sampled = sampled.view(N, C, H, W, v_sz)
        sampled = sampled.permute(0, 2, 3, 1, 4).reshape(N * H * W, C, v_sz)

        return sampled
    
    @staticmethod
    def compute_corr(v1:torch.Tensor, v2): #v1,v2: shape=[X, dim, size]
        assert v1.shape == v1.shape
        _,dim, _ = v1.shape 
        return torch.matmul(v1.transpose(1,2), v2) / torch.sqrt(torch.tensor(dim).float())

class Corr1DBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = Corr1DBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, w2) 
        
        self.corr_pyramid.append(corr.unsqueeze(-2))
        for i in range(self.num_levels-1):

            corr = F.avg_pool1d(corr, 2, stride=2)
            self.corr_pyramid.append(corr.unsqueeze(-2))

        
    def __call__(self, coords):
        #coords are 1D B, 1,H,W
        r = self.radius
        batch,_, h1, w1 = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            dx = torch.linspace(-r, r, 2*r+1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1) / 2**i
            delta_lvl = dx.view(1, 1, 2*r+1)
            coords_lvl = 2*(centroid_lvl + delta_lvl)/(w1-1)-1
            coords_lvl = torch.stack([coords_lvl, torch.zeros_like(coords_lvl)], dim=-1)

            corr = F.grid_sample(corr, coords_lvl, align_corners=True)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    def get_corr_pyramid(self):
        return self.corr_pyramid
    
    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        
        fmap1 = fmap1.permute(0,2,1,3)
        fmap2 = fmap2.permute(0,2,1,3)

        corr = torch.matmul(fmap1.transpose(2,3), fmap2)
        corr = corr.view(batch, ht, wd, 1, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())
  