"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register
#% Segmentation
import torch
from ..block import SegmentationHead


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        segmentation_head = None, #% Segmentation
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.segmentation_head = SegmentationHead(**segmentation_head) if segmentation_head is not None else None #% Segmentation

    # def forward(self, x, targets=None):
    #     x = self.backbone(x)
    #     x = self.encoder(x)
    #     x = self.decoder(x, targets)

    #     return x

    #% Segmentation
    def forward(self, x, targets=None):
        s = self.backbone(x)
        f = self.encoder(s)
        out = self.decoder(f, targets)

        if self.segmentation_head is not None:
            if self.training:
                outputs_masks = self.segmentation_head(f[0], out['hs'], x.shape[-2:])
                out['pred_masks'] = outputs_masks[-1]
                for a, b in zip(out['aux_outputs'], outputs_masks[:-1]):
                    a['pred_masks'] = b
                masks_enc = self.segmentation_head(f[0], [out['hs_enc'],], x.shape[-2:], skip_blocks=True)
                masks_enc = torch.cat(masks_enc, dim=1)
                out['enc_aux_outputs'][-1]['pred_masks'] = masks_enc
            else:
                outputs_masks = self.segmentation_head(f[0], out.pop('hs'), x.shape[-2:])
                out['pred_masks'] = outputs_masks[-1]

        return out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
