import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_utils import MambaBiDir
from .update_disp import DispUpdateBlock, SubPixCor
from .utils import coords_grid, upflow8

from .corr import CorrBlock
from .extractor import SmallEncoder, BasicEncoder
from .update_flow import BasicUpdateBlock

from utils.image_utils import ImagePadder


class EFlowDisp(nn.Module):
    def __init__(self, config, n_first_channels: int):
        super(EFlowDisp, self).__init__()

        self.image_padder = ImagePadder(min_size=32)

        self.height = config["height"]
        self.width = config["width"]

        self.hidden_dim = config["hidden_dim"]
        self.context_dim = config["context_dim"]

        self.corr_levels = config["corr_levels"]
        self.corr_radius = config["corr_radius"]
        self.disp_corr_levels = config["disp_corr_levels"]
        self.disp_corr_radius = config["disp_corr_radius"]

        self.fnet = SmallEncoder(output_dim=256, norm_fn="instance", dropout=0, n_first_channels=n_first_channels, use_mamba=config["feature_mamba"])
        self.cnet = BasicEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn="batch", dropout=0, n_first_channels=n_first_channels, use_mamba=config["context_mamba"])

        self.update_block = BasicUpdateBlock(corr_levels=config["corr_levels"], corr_radius=config["corr_radius"], hidden_dim=self.hidden_dim)
        self.disp_update_block = DispUpdateBlock(corr_levels=config["disp_corr_levels"], corr_radius=config["disp_corr_radius"], hidden_dim=self.hidden_dim)

        self.feat_4 = nn.Sequential(
            MambaBiDir(64, h=self.height // 4, w=self.width // 4),
            nn.Conv2d(64, 32, 1),
        )
        self.disp_cnet_mamba_x2 = MambaBiDir(dim=128, pos_enc=False)
        self.disp_cnet_conv2 = nn.Conv2d(128, self.hidden_dim + self.context_dim, kernel_size=1)

        self.disp_fnet_mamba_x2 = MambaBiDir(dim=96, pos_enc=False)
        self.disp_fnet_conv2 = nn.Conv2d(96, 256, kernel_size=1)

    def initialize_flow(self, x: torch.Tensor):
        n, _, h, w = x.shape
        coords0 = coords_grid(n, h // 8, w // 8).to(x.device)
        coords1 = coords_grid(n, h // 8, w // 8).to(x.device)
        return coords0, coords1

    def initialize_disp(self, x: torch.Tensor):
        n, _, h, w = x.shape
        coords0 = torch.arange(w // 8, device=x.device).repeat(n, h // 8, 1).float().unsqueeze(1)
        coords1 = torch.arange(w // 8, device=x.device).repeat(n, h // 8, 1).float().unsqueeze(1)
        return coords0, coords1

    @staticmethod
    def upsample_flow(flow: torch.Tensor, mask: torch.Tensor):
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, 8 * h, 8 * w)

    @staticmethod
    def upsample_disp(disp: torch.Tensor, mask: torch.Tensor):
        n, _, h, w = disp.shape
        mask = mask.view(n, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(8 * disp, [3, 3], padding=1)
        up_disp = up_disp.view(n, 1, 9, 1, 1, h, w)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(n, 1, 8 * h, 8 * w)

    def forward(
        self,
        imageL0,
        imageL1,
        imageR0=None,
        iters: int = 12,
        disp_iters: int = 8,
        flow_init=None,
        disp_init=None,
        recurr: bool = False,
        flow_only: bool = False,
    ):
        imageL0 = self.image_padder.pad(imageL0).contiguous()
        imageL1 = self.image_padder.pad(imageL1).contiguous()

        if not flow_only:
            if imageR0 is None:
                raise ValueError("imageR0 must be provided when flow_only=False.")
            imageR0 = self.image_padder.pad(imageR0).contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        if flow_only:
            f_out = self.fnet([imageL0, imageL1], return_inter=False)
            fmapL0, fmapL1 = f_out
        else:
            f_sc4, f_disp, f_out = self.fnet([imageL0, imageL1, imageR0], return_inter=True)
            fmapL0, fmapL1, _ = f_out

            f_disp = self.disp_fnet_conv2(
                self.disp_fnet_mamba_x2(torch.cat([f_disp[0], f_disp[2]], dim=0))
            )
            fdispL0, fdispR0 = torch.split(f_disp, fmapL0.shape[0], dim=0)

            ff4 = self.feat_4(torch.cat([f_sc4[0], f_sc4[2]], dim=0))
            fmapL_sc4, fmapR_sc4 = torch.split(ff4, fmapL0.shape[0], dim=0)

        corr_fn = CorrBlock(
            fmapL0,
            fmapL1,
            num_levels=self.corr_levels,
            radius=self.corr_radius,
        )

        cnet = self.cnet(imageL1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(imageL1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0

            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow

            flow_up = upflow8(coords1 - coords0) if up_mask is None else self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(self.image_padder.unpad(flow_up))

        if flow_only:
            return coords1 - coords0, flow_predictions if recurr else flow_predictions[-1]
        
        corr_disp = SubPixCor(fdispL0, fdispR0, num_levels=self.disp_corr_levels, radius=self.disp_corr_radius)
        corr_disp2 = SubPixCor(fmapL_sc4, fmapR_sc4, num_levels=self.disp_corr_levels, radius=self.disp_corr_radius)

        cnet_disp = self.cnet(imageL0, return_inter_only=True)
        cnet_disp = self.disp_cnet_conv2(self.disp_cnet_mamba_x2(cnet_disp))
        disp_net, disp_inp = torch.split(cnet_disp, [hdim, cdim], dim=1)
        disp_net = torch.tanh(disp_net)
        disp_inp = torch.relu(disp_inp)

        coordsL, coordsR = self.initialize_disp(imageL0)
        coordsR_sc4 = (
            torch.arange(fmapL_sc4.shape[-1], device=fmapL_sc4.device)
            .repeat(fmapL_sc4.shape[0], fmapL_sc4.shape[-2], 1).float().unsqueeze(1)
        )

        if disp_init is not None:
            coordsR = coordsL - F.interpolate(disp_init, scale_factor=(0.5, 0.5), mode="area") / 2.0
            coordsR_sc4 = coordsR_sc4 - disp_init
        else:
            coordsR = coordsR - self.disp_corr_radius + 1
            coordsR_sc4 = coordsR_sc4 - self.disp_corr_radius

        disp_predictions = []
        for _ in range(disp_iters):
            coordsR = coordsR.detach()
            corr1 = corr_disp(coordsR)
            corr2 = corr_disp2(coordsR_sc4.detach())
            corr = torch.cat([corr1, F.pixel_unshuffle(corr2, 2)], dim=1)

            disp = coordsL - coordsR
            disp_net, disp_up_mask, delta_disp = self.disp_update_block(disp_net, corr, disp_inp, disp)

            coordsR = coordsR - delta_disp
            disp_up = self.upsample_disp(coordsL - coordsR, disp_up_mask)
            coordsR_sc4 = coordsR_sc4 - F.interpolate(disp_up, scale_factor=0.25, mode="area") / 4.0

            disp_predictions.append(self.image_padder.unpad(disp_up))

        if recurr:
            return coords1 - coords0, flow_predictions, disp_predictions

        return flow_predictions, disp_predictions