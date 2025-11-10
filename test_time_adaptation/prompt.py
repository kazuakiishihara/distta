from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

import engine_fun.regisry as regisry

class CrossAttention(nn.Module):
    def __init__(self, C, K):
        super(CrossAttention, self).__init__()
        # Q, K, V mapping

        self.query = nn.Conv3d(C, C, kernel_size=3, stride=1, padding=1, groups=C)
        self.key = nn.Conv3d(K, K, kernel_size=3, stride=1, padding=1, groups=K)
        self.value = nn.Conv3d(K, K, kernel_size=3, stride=1, padding=1, groups=K)
    
    def forward(self, x_q, x_kv):
        query = self.query(x_q)
        key = self.key(x_kv)
        value = self.value(x_kv)

        batch_size, channels, depth, height, width = query.size()
        batch_size, num_prompt, depth, height, width = key.size()
        query = query.view(batch_size, channels, -1)
        key = key.view(batch_size,num_prompt, -1).permute(0, 2, 1)
        value = value.view(batch_size, num_prompt, -1)

        # calculate attention weights with scaled dot-product attention
        attn_weights = torch.matmul(query, key) / (num_prompt ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1) # Shape in attention map: [C, K]

        # sparsity atten
        num_non_zero = int(0.1 * attn_weights.size(1)) # Activate 10% of shape template only
        sorted_values, sorted_indices = torch.sort(attn_weights, descending=True, dim=1)
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(1, sorted_indices[:, :num_non_zero], 1)
        attn_weights = attn_weights * mask

        # calculate attentive output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.view(batch_size, channels, depth, height, width)

        return x_q + attn_output, attn_weights


##########################################################################
class TransMorph_SPTTA(nn.Module):
    def __init__(self, pretrained_path, patch_size=(192, 224, 160)):
        super().__init__()
        data_prompt = torch.zeros(1, 768//2, patch_size[0]//32, patch_size[1]//32, patch_size[2]//32)
        self.data_prompt = nn.Parameter(data_prompt)
        self.prompt2feature = CrossAttention(C=768//2, K=768//2)
        self.data_adaptor = nn.Sequential(nn.Conv3d(1, 32, 1, stride=1, padding=0), 
                                        nn.InstanceNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(32, 1, 1, stride=1, padding=0))
        self.transmorph = regisry.build_model('TransMorph_DS', (192, 224, 160))
        pretrained_params = torch.load(pretrained_path, weights_only=False)['state_dict']
        self.transmorph.load_state_dict(pretrained_params)
        for name, param in self.transmorph.named_parameters():
            if 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, inputs):
        source, tar = inputs # moving, fixed
        perturbation = self.data_adaptor(tar) # NOTE For atlas-based registration
        tar = tar + perturbation
        x = torch.cat((source, tar), dim=1)
        if self.transmorph.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.transmorph.avg_pool(x)
            f4 = self.transmorph.c1(x_s1)
            f5 = self.transmorph.c2(x_s0)
        else:
            f4 = None
            f5 = None
        
        feats_m = self.transmorph.transformer(source)
        feats_f = self.transmorph.transformer(tar)

        adapted_feats_f, attn_weights = self.prompt2feature(x_q=feats_f[-1], x_kv=self.data_prompt)

        if self.transmorph.if_transskip:
            f1 = torch.cat((feats_m[-2], feats_f[-2]), dim=1)
            f2 = torch.cat((feats_m[-3], feats_f[-3]), dim=1)
            f3 = torch.cat((feats_m[-4], feats_f[-4]), dim=1)
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.transmorph.up0(torch.cat((feats_m[-1], adapted_feats_f), dim=1), f1)
        x = self.transmorph.up1(x, f2)
        x = self.transmorph.up2(x, f3)
        x = self.transmorph.up3(x, f4)
        x = self.transmorph.up4(x, f5)
        flow = self.transmorph.reg_head(x)
        out = self.transmorph.spatial_trans(source, flow)
        return out, flow

if __name__ == '__main__':
    pretrained_path = './logs/ixi_ar/Oct14-205009_TransMorph/model_wts/TransMorph_dsc0.7417_epoch42.pth.tar'
    model = TransMorph_SPTTA(pretrained_path)
    output = model(
                (torch.rand((1,1,192,224,160)), torch.rand((1,1,192,224,160)))
                )
