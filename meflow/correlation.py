import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler


class Correlation1D:
    def __init__(self, feature1, feature2,
                 radius=32,
                 x_correlation=False,
                 ):
        self.radius = radius
        self.x_correlation = x_correlation

        if self.x_correlation:
            self.corr = self.corr_x(feature1, feature2)  # [B*H*W, 1, 1, W]
        else:
            self.corr = self.corr_y(feature1, feature2)  # [B*H*W, 1, H, 1]

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
        b, h, w = coords.shape[:3]

        if self.x_correlation:
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.zeros_like(dx)
            delta_x = torch.stack((dx, dy), dim=-1).to(coords.device)  # [2r+1, 2]

            coords_x = coords[:, :, :, 0]  # [B, H, W]
            coords_x = torch.stack((coords_x, torch.zeros_like(coords_x)), dim=-1)  # [B, H, W, 2]

            centroid_x = coords_x.view(b * h * w, 1, 1, 2)  # [B*H*W, 1, 1, 2]
            coords_x = centroid_x + delta_x  # [B*H*W, 1, 2r+1, 2]

            coords_x = 2 * coords_x / (w - 1) - 1  # [-1, 1], y is always 0

            corr_x = F.grid_sample(self.corr, coords_x, mode='bilinear',
                                   align_corners=True)  # [B*H*W, G, 1, 2r+1]

            corr_x = corr_x.view(b, h, w, -1)  # [B, H, W, (2r+1)*G]
            corr_x = corr_x.permute(0, 3, 1, 2).contiguous()  # [B, (2r+1)*G, H, W]
            return corr_x
        else:  # y correlation
            dy = torch.linspace(-r, r, 2 * r + 1)
            dx = torch.zeros_like(dy)
            delta_y = torch.stack((dx, dy), dim=-1).to(coords.device)  # [2r+1, 2]
            delta_y = delta_y.unsqueeze(1).unsqueeze(0)  # [1, 2r+1, 1, 2]

            coords_y = coords[:, :, :, 1]  # [B, H, W]
            coords_y = torch.stack((torch.zeros_like(coords_y), coords_y), dim=-1)  # [B, H, W, 2]

            centroid_y = coords_y.view(b * h * w, 1, 1, 2)  # [B*H*W, 1, 1, 2]
            coords_y = centroid_y + delta_y  # [B*H*W, 2r+1, 1, 2]

            coords_y = 2 * coords_y / (h - 1) - 1  # [-1, 1], x is always 0

            corr_y = F.grid_sample(self.corr, coords_y, mode='bilinear',
                                   align_corners=True)  # [B*H*W, G, 2r+1, 1]

            corr_y = corr_y.view(b, h, w, -1)  # [B, H, W, (2r+1)*G]
            corr_y = corr_y.permute(0, 3, 1, 2).contiguous()  # [B, (2r+1)*G, H, W]

            return corr_y

    def corr_x(self, feature1, feature2):
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c ** 0.5

        # x direction
        feature1 = feature1.permute(0, 2, 3, 1)  # [B, H, W, C]
        feature2 = feature2.permute(0, 2, 1, 3)  # [B, H, C, W]
        corr = torch.matmul(feature1, feature2)  # [B, H, W, W]

        corr = corr.unsqueeze(3).unsqueeze(3)  # [B, H, W, 1, 1, W]
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)  # [B*H*W, 1, 1, W]

        return corr

    def corr_y(self, feature1, feature2):
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c ** 0.5

        # y direction
        feature1 = feature1.permute(0, 3, 2, 1)  # [B, W, H, C]
        feature2 = feature2.permute(0, 3, 1, 2)  # [B, W, C, H]
        corr = torch.matmul(feature1, feature2)  # [B, W, H, H]

        corr = corr.permute(0, 2, 1, 3).contiguous().view(b, h, w, 1, h, 1)  # [B, H, W, 1, H, 1]
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)  # [B*H*W, 1, H, 1]

        return corr

class DeformedCorr1D:
    def __init__(self, fmap1, fmap2, flow_init, radius=32, v_corr=True):

        self.radius = radius
        self.v_corr = v_corr

        if self.v_corr: 
            self.corr = self.corr_v(fmap1, fmap2, flow_init) # [B*H*W, 1, 1, H]
        else:
            self.corr = self.corr_h(fmap1, fmap2, flow_init) # [B*H*W, 1, 1, W]

    def __call__(self, coords):
        r = self.radius
        b, h, w = coords.shape
        if self.v_corr:
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(b*h*w, 1, 1, 1)
            x0 = 2 * x0 / (h-1) -1
            coords_lvl = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            corr_v = F.grid_sample(self.corr, coords_lvl, mode='bilinear',
                                   align_corners=True)  # [B*H*W, 1, 1, 2r+1]
            corr_v = corr_v.view(b, h, w, -1)
            return corr_v.permute(0, 3, 1, 2).contiguous().float()

        else:
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(b*h*w, 1, 1, 1)
            x0 = 2 * x0 / (w-1) -1
            coords_lvl = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            corr_h = F.grid_sample(self.corr, coords_lvl, mode='bilinear',
                                   align_corners=True)  # [B*H*W, 1, 1, 2r+1]
            corr_h = corr_h.view(b, h, w, -1)
            return corr_h.permute(0, 3, 1, 2).contiguous().float()


    def corr_v(self, fmap1, fmap2, flow_init):
        b, c, h, w = fmap1.shape # [B, C, H, W]
        scale_factor = c ** 0.5
        coords_x = torch.arange(0, w).view(1, 1, w).to(flow_init.device)  # [1, 1, W]

        coords_x = coords_x + flow_init.view(b, h, w)
        coords_x = 2 * coords_x / (w-1) -1
        coords = torch.stack((coords_x, torch.zeros_like(coords_x)), dim=-1) # [B, H, W, 2]

        corr_v = fmap1.new_zeros([b, h, h, w])
        for i in range(c):
            deform_fmap2 = F.grid_sample(fmap2[:, i, :, None, :], coords, mode='bilinear',
                                   align_corners=True)  # [B, H, H, W]
            corr_v += fmap1[:, i, None, :, :] * deform_fmap2 # [B, H, H, W]
        return corr_v.permute(0, 2, 3, 1).reshape(b*h*w, 1, 1, h) / scale_factor

    def corr_h(self, fmap1, fmap2, flow_init):
        b, c, h, w = fmap1.shape # [B, C, H, W]
        scale_factor = c ** 0.5
        coords_y = torch.arange(0, h).view(1, h, 1).to(flow_init.device)  # [1, 1, W]

        coords_y = coords_y + flow_init.view(b, h, w)
        coords_y = 2 * coords_y / (h-1) -1
        coords = torch.stack((torch.zeros_like(coords_y), coords_y), dim=-1) # [B, H, W, 2]
        fmap2 = fmap2.permute(0, 1, 3, 2) # [B, C, W, H]
        corr_h = fmap1.new_zeros([b, w, h, w])
        for i in range(c):
            deform_fmap2 = F.grid_sample(fmap2[:, i, :, :, None], coords, mode='bilinear',
                                   align_corners=True)  # [B, W, H, W]
            corr_h += fmap1[:, i, None, :, :] * deform_fmap2 # [B, W, W, H]
        return corr_h.permute(0, 2, 3, 1).reshape(b*h*w, 1, 1, w) / scale_factor

class AlternateCorr1D:
    def __init__(self, fmap1, fmap2, fmap2_d1, fmap2_d2, radius=4, h_corr=True):
        self.radius = radius
        self.h_corr = h_corr

        if self.h_corr:
            self.fmap1 = fmap1
            self.fmap2 = fmap2
            self.fmap2_d1 = fmap2_d1
            self.fmap2_d2 = fmap2_d2
            b, c, h, w = self.fmap1.shape
            self.c = c
        else:
            self.fmap1 = fmap1
            self.fmap2 = fmap2
            self.fmap2_d1 = fmap2_d1
            self.fmap2_d2 = fmap2_d2
            b, c, h, w = self.fmap1.shape
            self.c = c

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        b, h, w, _ = coords.shape

        if self.h_corr:
            dx = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack((dx, torch.zeros_like(dx)), dim=-1).to(coords.device)  # [2r+1, 2]
        else:
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack((torch.zeros_like(dy), dy), dim=-1).to(coords.device)  # [2r+1, 2]
        coords_lvl = coords.view(b, 1, h, w, 2) + delta.view(1, 2*r+1, 1, 1, 2)            
        warped_fmap2 = bilinear_sampler(self.fmap2, coords_lvl.reshape(b, -1, w, 2)) # [b, c, (2*r+1)*h, w]
        warped_fmap2 = warped_fmap2.view(b, self.c, 2*r+1, h, w)

        delta_d = torch.cat((delta[0:2], delta[-2:]), dim=0)
        coords_lvl_d1 = coords.view(b, 1, h, w, 2)/2 + delta_d.view(1, 4, 1, 1, 2)
        warped_fmap2_d1 = bilinear_sampler(self.fmap2_d1, coords_lvl_d1.reshape(b, -1, w, 2)) # [b, c, 4*h, w]
        warped_fmap2_d1 = warped_fmap2_d1.view(b, self.c, 4, h, w)

        coords_lvl_d2 = coords.view(b, 1, h, w, 2)/4 + delta_d.view(1, 4, 1, 1, 2)
        warped_fmap2_d2 = bilinear_sampler(self.fmap2_d2, coords_lvl_d2.reshape(b, -1, w, 2)) # [b, c, 4*h, w]
        warped_fmap2_d2 = warped_fmap2_d2.view(b, self.c, 4, h, w)

        warped_fmap2 = torch.cat((warped_fmap2_d2[:,:,0:2], warped_fmap2_d1[:,:,0:2], warped_fmap2, warped_fmap2_d1[:,:,-2:], warped_fmap2_d2[:,:,-2:]), dim=2)
        corr = (self.fmap1[:, :, None, :, :] * warped_fmap2).sum(dim=1)
        return corr / (self.c ** 0.5)
