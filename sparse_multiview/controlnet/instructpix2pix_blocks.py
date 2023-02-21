import torch
import torch.nn as nn

class InstructPix2PixUNet(nn.Module):
    def __init__(self, opts, unet):
        super().__init__()
        self.unet = unet
        # To avoid StableDiffusionPipeline errors
        self.in_channels = self.unet.in_channels

        self.variant = opts.variant
        self.pose_embed_dim = opts.pose_embed_dim
        self.pose_mode = opts.pose_mode
        self.patch_unet()

        if self.variant ==  'x_y_theta_phi_nn':
            self.proj = nn.Linear(4*self.pose_embed_dim, 2*self.pose_embed_dim)

    @property
    def config(self):
        return self.unet.config
    @property
    def device(self):
        return self.unet.device
    @property
    def dtype(self):
        return self.unet.dtype

    def embed_pose(self, pose, dim):
        # from TARS
        assert(dim % 2 == 0)
        # pose: [B, ..., S]
        L, shape, last_dim = dim//2, pose.shape, pose.shape[-1]
        freq = torch.arange(L, dtype=torch.float32, device=pose.device) / L
        freq = 1./10000**freq  # (D/2,) # from Attention is all you need
        spectrum = pose[..., None] * freq  # [B,...,S,L]
        pos_embed = torch.cat([spectrum.sin(), spectrum.cos()], dim=-1).view(*shape[:-1], 2*last_dim*L)  # [B,...,DS]
        return pos_embed

    def encode_pose(self, batch):
        #TODO: normalize for positional encoding??
        theta_phi = batch['pose']
        device = theta_phi.device
        match self.pose_mode:
            case 'fourier':
                embed_theta_phi = self.embed_pose(theta_phi, self.pose_embed_dim).view(-1, 2*self.pose_embed_dim, 1, 1).repeat(1, 1, 64, 64)
                bsz = theta_phi.shape[0]
                x_y = torch.stack(torch.meshgrid(torch.arange(64, device=device), torch.arange(64, device=device), indexing='ij'), axis=-1) # see https://pytorch.org/docs/stable/generated/torch.meshgrid.html
                embed_x_y = self.embed_pose(x_y, self.pose_embed_dim).view(1, 64, 64, 2*self.pose_embed_dim).repeat(bsz, 1, 1, 1).permute((0, 3, 1, 2))
            case "learn":
                raise "Unimplemented"
            case _: 
                raise "Unrecognized pose_mode"

        match self.variant:
            case 'no_pose':
                embeds = None
            case 'theta_phi':
                embeds = embed_x_y
            case 'x_y_theta_phi':
                embeds = torch.cat([embed_x_y, embed_theta_phi], axis=1)
            case 'x_y_theta_phi_nn':
                embeds = self.proj(torch.cat([embed_x_y, embed_theta_phi], axis=1).permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            case _:
                raise "Unrecognized variant"
        return embeds

    def patch_unet(self):
        with torch.no_grad():
            match self.variant:
                case 'no_pose':
                    extra_channels = 4
                case 'theta_phi':
                    extra_channels = 24
                case 'x_y_theta_phi':
                    extra_channels = 44
                case 'x_y_theta_phi_nn':
                    extra_channels = 24
                case _:
                    raise "Unrecognized variant"

            kernel_size, in_channels, block_out_channels = 3, 4, (320, 640, 1280, 1280)
            conv_old = self.unet.conv_in
            has_bias = conv_old.bias is not None
            conv_new = nn.Conv2d(in_channels + extra_channels, block_out_channels[0], kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=has_bias)

            weight = torch.zeros_like(conv_new.weight)
            weight[:,:in_channels,:,:] = conv_old.weight.detach().clone()
            conv_new.weight.copy_(weight)
            if has_bias:
                conv_new.bias.copy_(conv_old.bias.detach().clone())

            self.unet.conv_in = conv_new
            self.unet.conv_in.requires_grad = True
            print(f'Patched unet conv_in: {self.unet.conv_in.weight.shape}')
            
    def forward(self, noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs=None):
        source_latents = cross_attention_kwargs['source_latents']
        pose_embeds = cross_attention_kwargs['pose_embeds']
        noisy_latents = torch.cat([noisy_latents, source_latents, pose_embeds] if pose_embeds is not None else [noisy_latents, source_latents], dim=1) #.repeat(noisy_latents.shape[0], 1, 1, 1)
        return self.unet.forward(noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs)
