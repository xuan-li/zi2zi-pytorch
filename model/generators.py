import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128, norm_layer=nn.BatchNorm2d, use_dropout=False, is_training=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, embedding_dim=embedding_dim)  # add the innermost layer
        for _ in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.encoder = self.model.encoder
        self.embedder = nn.Embedding(embedding_num, embedding_dim)
        self.is_training = is_training
    def forward(self, style_or_label, x):
        """Standard forward"""
        if self.is_training:
            return self.model(self.embedder(style_or_label), x)
        else:
            return self.model(style_or_label, x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, embedding_dim=128, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            encoder = down + [submodule.encoder]

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            encoder = down

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            encoder = down + [submodule.encoder]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]
        
        self.submodule = submodule
        self.encoder = nn.Sequential(*encoder)
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, style, x):
        if self.innermost:
            encode = self.down(x)
            enc = torch.cat([style.view(style.shape[0], style.shape[1], 1, 1), encode], 1)
            dec = self.up(enc)
            return torch.cat([x, dec], 1), encode.view(x.shape[0], -1)
        elif self.outermost:
            enc = self.down(x)
            sub, encode = self.submodule(style, enc)
            dec = self.up(sub)
            return dec, encode
        else:   # add skip connections
            enc = self.down(x)
            sub, encode = self.submodule(style, enc)
            dec = self.up(sub)
            return torch.cat([x, dec], 1), encode