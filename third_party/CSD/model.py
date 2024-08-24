import torch
import torch.nn as nn
from CSD import clip
import copy
from torch.autograd import Function


from .utils import convert_weights_float


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


## taken from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def init_weights(m): # TODO: do we need init for layernorm?
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=1e-6)


class CSD_CLIP(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='vit_large', content_proj_head='default'):
        super(CSD_CLIP, self).__init__()
        self.content_proj_head = content_proj_head
        if name == 'vit_large':
            clipmodel, _ = clip.load("ViT-L/14")
            self.backbone = clipmodel.visual
            self.embedding_dim = 1024
        elif name == 'vit_base':
            clipmodel, _ = clip.load("ViT-B/16")
            self.backbone = clipmodel.visual
            self.embedding_dim = 768 
            self.feat_dim = 512
        else:
            raise Exception('This model is not implemented')

        convert_weights_float(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        if content_proj_head == 'custom':
            self.last_layer_content = ProjectionHead(self.embedding_dim,self.feat_dim)
            self.last_layer_content.apply(init_weights)
        else:
            self.last_layer_content = copy.deepcopy(self.backbone.proj)

        self.backbone.proj = None

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, input_data, alpha=None):
        
        feature = self.backbone(input_data)

        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
        else:
            reverse_feature = feature

        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)

        # if alpha is not None:
        if self.content_proj_head == 'custom':
            content_output =  self.last_layer_content(reverse_feature)
        else:
            content_output = reverse_feature @ self.last_layer_content
        content_output = nn.functional.normalize(content_output, dim=1, p=2)
        return feature, content_output, style_output
