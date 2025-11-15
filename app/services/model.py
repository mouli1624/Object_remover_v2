import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer,
    CLIPPreTrainedModel,
    CLIPModel,
)


class CLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
        cache_dir
    ):
        model = CLIPModel.from_pretrained(
            global_model_name_or_path,
            subfolder="image_prompt_encoder", 
            cache_dir=cache_dir
        )
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return CLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    def forward(self, object_pixel_values):
        b, c, h, w = object_pixel_values.shape

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode="bilinear", antialias=True
            )

        object_pixel_values = self.vision_processor(object_pixel_values)
        object_embeds = self.vision_model(object_pixel_values)[1]
        object_embeds = self.visual_projection(object_embeds)
        object_embeds = object_embeds.view(b, 1, -1)
        return object_embeds
    
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x

class PostfuseModule(nn.Module):
    def __init__(self, embed_dim, embed_dim_img):
        super().__init__()
        self.mlp1 = MLP(embed_dim_img, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def fuse_fn(self, object_embeds):
        text_object_embeds = self.mlp1(object_embeds)
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        fuse_index,
    ) -> torch.Tensor:
        text_object_embed = self.fuse_fn(object_embeds)
        text_embeds_new = text_embeds.clone()
        text_embeds_new[:, fuse_index, :] = text_object_embed.squeeze(1)

        return text_embeds_new