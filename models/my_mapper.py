import torch
from torch.nn import Module
from models.stylegan2.model import EqualLinear, PixelNorm

def Mapper(latent_dim=512):
    layers = [PixelNorm()]
    for i in range(4):
        layers.append(
            EqualLinear(
                latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
            )
        )

    return torch.nn.Sequential(*layers)

class LevelsMapper(Module):

    def __init__(self):
        super(LevelsMapper, self).__init__()

        self.course_mapping = Mapper()
        self.medium_mapping = Mapper()
        self.fine_mapping = Mapper()

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        x_coarse = self.course_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

def load_mapper(device='cpu', path='./weights/mapper_curly_hairstyle.pt'):
    mapper = LevelsMapper().to(device)
    mapper.load_state_dict(torch.load(path), strict=False)
    return mapper