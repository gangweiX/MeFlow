import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import FeatureNet, ContextNet
from .local_attention import Attention1D
from .position import PositionEmbeddingSine
from .correlation import AlternateCorr1D
from .update import BasicUpdateBlock
from utils.utils import coords_grid
import time


class Model(nn.Module):
    def __init__(self,
                 downsample_factor=8,
                 feature_channels=128,
                 hidden_dim=128,
                 context_dim=128,
                 corr_radius=4,
                 mixed_precision=False,
                 **kwargs,
                 ):
        super(Model, self).__init__()

        self.downsample_factor = downsample_factor
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

        # feature network, context network, and update block
        self.fnet = FeatureNet(output_dim=feature_channels, norm_fn='instance',
                                 )

        self.cnet = ContextNet(output_dim=hidden_dim + context_dim, norm_fn='batch',
                                 )

        # 1D attention
        corr_channels = (2 * (corr_radius + 4) + 1) * 2

        self.attn_h = Attention1D(feature_channels,
                                  h_attention=True,
                                  r=4
                                  )

        self.attn_v = Attention1D(feature_channels,
                                  h_attention=False,
                                  r=4
                                  )

        # Update block
        self.update_block = BasicUpdateBlock(corr_channels=corr_channels,
                                             hidden_dim=hidden_dim,
                                             context_dim=context_dim,
                                             downsample_factor=downsample_factor,
                                             )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, downsample=None):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)
        coords1 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def learned_upflow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, self.downsample_factor, self.downsample_factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.downsample_factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, self.downsample_factor * h, self.downsample_factor * w)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False,
                ):
        """ Estimate optical flow between pair of frames """

        # torch.cuda.reset_max_memory_allocated()

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap2_attn_h = self.attn_h(fmap2)
        fmap2_attn_v = self.attn_v(fmap2) 


        fmap2_d1 = F.avg_pool2d(fmap2, 2, stride=2)
        fmap2_attn_h_d1 = self.attn_h(fmap2_d1)
        fmap2_attn_v_d1 = self.attn_v(fmap2_d1)

        fmap2_d2 = F.avg_pool2d(fmap2_d1, 2, stride=2)
        fmap2_attn_h_d2 = self.attn_h(fmap2_d2)
        fmap2_attn_v_d2 = self.attn_v(fmap2_d2)

        corr_fn_h = AlternateCorr1D(fmap1, fmap2_attn_v, fmap2_attn_v_d1, fmap2_attn_v_d2, radius=self.corr_radius, h_corr=True)
        corr_fn_v = AlternateCorr1D(fmap1, fmap2_attn_h, fmap2_attn_h_d1, fmap2_attn_h_d2, radius=self.corr_radius, h_corr=False)

        # run the context network
        cnet = self.cnet(image1)  # list of feature pyramid, low scale to high scale

        hdim = self.hidden_dim
        cdim = self.context_dim
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)  # 1/8 resolution or 1/4

        if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()  # stop gradient
            flow = coords1 - coords0
            corr_h = corr_fn_h(coords1) # index 1D correlation volume
            corr_v = corr_fn_v(coords1) # index 1D correlation volume
            corr = torch.cat((corr_h, corr_v), dim=1)

            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow,
                                                         upsample=not test_mode or itr == iters - 1,
                                                         )

            coords1 = coords1 + delta_flow

            if test_mode:
                torch.cuda.empty_cache()
                # only upsample the last iteration
                if itr == iters - 1:
                    flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                    # print('Max Allocated:', round(torch.cuda.max_memory_allocated(0)/1024**3,2), 'GB')

                    return coords1 - coords0, flow_up
            else:
                # upsample predictions
                flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                flow_predictions.append(flow_up)

        return flow_predictions

def build_model(args):
    return Model(downsample_factor=args.downsample_factor,
                 feature_channels=args.feature_channels,
                 corr_radius=args.corr_radius,
                 hidden_dim=args.hidden_dim,
                 context_dim=args.context_dim,
                 mixed_precision=args.mixed_precision,
                 )
