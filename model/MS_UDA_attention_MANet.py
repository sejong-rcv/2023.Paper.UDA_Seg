
import torch.nn as nn

from .MANet_encoder import MANet_encoder
from .discriminator import fin_Discriminator
from .MANet_decoder import MANet_decoder
from utils.initial_utils import init_net
from utils.utils import find_norm


class MS_UDA_attention_MANet(nn.Module):
    ## resnet pretrained version

    def __init__(self, args, num_layers=50):
        super(MS_UDA_attention_MANet, self).__init__()

        self.num_layers = num_layers
        self.inplanes = 2048

        norm = find_norm(args.norm)

        #we initialize the network and wrap with dataparallel/single gpu.
        # encoders(rgb) use pretrained weight
        # decoder uses xavier uniform
        # discriminator is not initialized
        
        self.net_G_rgb = init_net(MANet_encoder(sensor = 'rgb'), init_type='normal', #원래 init_type=False
                                  net_type='encoder_rgb', gpu=args.gpus,
                                  init_gain=args.init_gain)
        self.net_G_thermal = init_net(MANet_encoder(sensor = 'thermal'), init_type='normal', #원래 init_type=False
                                      net_type='encoder_th', gpu=args.gpus)
        self.decoder = init_net(MANet_decoder(args.num_classes, args.model), init_type=args.init_type,
                                net_type='decoder', gpu=args.gpus)
        self.fin_D = init_net(fin_Discriminator(args.num_classes, norm_layer=norm), init_type=args.init_type,
                              net_type='discriminator', gpu=args.gpus)

        #Following RTFNet, we use pretrained encoder and initialize decoder with xavier method.
        #Following patchGAN, we initialize discriminator using xavier method with 0.02 as initial gain


