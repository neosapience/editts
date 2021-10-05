# Modified from original Grad-TTS code: https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS

import math
import random

import torch
from torch.nn import functional as F

from text import sequence_to_text
from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
    fix_len_compatibility_text_edit,
)
from model.commons import shift_mel

class GradTTS(BaseModule):
    def __init__(
        self,
        n_vocab,
        n_enc_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_enc_layers,
        enc_kernel,
        enc_dropout,
        window_size,
        n_feats,
        dec_dim,
        beta_min,
        beta_max,
        pe_scale,
    ):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.encoder = TextEncoder(
            n_vocab,
            n_feats,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
        )
        self.decoder = Diffusion(n_feats, dec_dim, beta_min, beta_max, pe_scale)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        # Compute loss of score-based decoder
        diff_loss = self.decoder.compute_loss(y, y_mask, mu_y)

        return dur_loss, prior_loss, diff_loss

    @torch.no_grad()
    def edit_pitch(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        length_scale=1.0,
        soften_mask=True,
        n_soften=16,
        emphases=None,
        direction='up'
    ):
        x, x_lengths = self.relocate_input([x, x_lengths])

        mu_x, logw, x_mask = self.encoder(x, x_lengths)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
    
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        eps = torch.randn_like(mu_y, device=mu_y.device) / temperature
        z = mu_y + eps

        encoder_outputs = mu_y[:, :, :y_max_length]

        mu_x_edit = mu_x.clone()
        mask_edit = torch.zeros_like(mu_x[:, :1, :])
        for j, (start, end) in enumerate(emphases):
            mask_edit[:, :, start:end] = 1
            mu_x_edit[:, :, start:end] = shift_mel(mu_x_edit[:, :, start:end], direction=direction)

        mu_y_edit = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x_edit.transpose(1, 2)
        )        
        mask_edit = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mask_edit.transpose(1, 2)
        )

        mu_y_edit = mu_y_edit.transpose(1, 2)
        mask_edit = mask_edit.transpose(1, 2) # [B, 1, T]
        mask_edit[:, :, y_max_length:] = mask_edit[:, :, y_max_length-1] # for soften_mask

        z_edit = mu_y_edit + eps

        dec_out, dec_edit = self.decoder.double_forward_pitch(
            z, z_edit, mu_y, mu_y_edit, y_mask, mask_edit, n_timesteps, stoc, soften_mask, n_soften
        )
    
        # For baseline
        emphases_expanded = []
        attn = attn.squeeze()
        for start, end in emphases:
            i = attn[:start].sum().long().item() if start > 0 else 0
            j = attn[:end].sum().long().item()
            itv = [i, j]
            emphases_expanded.append(itv)

        dec_out = dec_out[:, :, :y_max_length]
        dec_baseline = dec_out.clone()
        for start, end in emphases_expanded:
            dec_baseline[:, :, start:end] = shift_mel(dec_baseline[:, :, start:end], direction=direction)
        dec_edit = dec_edit[:, :, :y_max_length]

        return dec_out, dec_baseline, dec_edit
    
    @torch.no_grad()
    def edit_content(
        self,
        x1,
        x2,
        x1_lengths,
        x2_lengths,
        emphases1,
        emphases2,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        length_scale=1.0,
        soften_mask=True,
        n_soften_text=9,
        n_soften=16,
        amax=0.9,
        amin=0.1
    ):
        def _process_input(x, x_lengths):
            x, x_lengths = self.relocate_input([x, x_lengths])

            mu_x, logw, x_mask = self.encoder(x, x_lengths)
            w = torch.exp(logw) * x_mask
            w_ceil = torch.ceil(w) * length_scale
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = int(y_lengths.max())
            y_max_length_ = fix_len_compatibility(y_max_length)

            y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2) # [1, n_mels, T]
            return mu_y, attn, y_mask, y_max_length, y_lengths
        
        def _soften_juntions(y_edit, y1, y2, y_edit_lengths, y1_lengths, y2_lengths, i1, j1, i2 ,j2):
            for n in range(1, n_soften_text + 1):
                alpha = (amax - amin) * (n_soften_text - n) / (n_soften_text - 1) + amin
                if i1 - n >= 0 and i2 - n >= 0:
                    y_edit[:, :, i1-n] = (1-alpha) * y1[:, :, i1-n] + alpha * y2[:, :, i2-n]
                if i1 + (j2 - i2) + n < y_edit_lengths and j1 + (n-1) < y1_lengths and j2 + (n-1) < y2_lengths:
                    y_edit[:, :, i1 + (j2 - i2) + (n-1)] = (1-alpha) * y1[:, :, j1 + (n-1)] + alpha * y2[:, :, j2 + (n-1)]
            return y_edit

        assert len(x1) == 1 and len(x2) == 1
        assert emphases1 is not None and emphases2 is not None
        assert len(emphases1) == 1 and len(emphases2) == 1

        mu_y1, attn1, y1_mask, y1_max_length, y1_lengths = _process_input(x1, x1_lengths) # mu_y1: [1, n_mels, T]
        mu_y2, attn2, y2_mask, y2_max_length, y2_lengths = _process_input(x2, x2_lengths) # mu_y2: [1, n_mels, T]

        attn1 = attn1.squeeze() # [N, T]
        attn2 = attn2.squeeze() # [N, T]

        i1 = attn1[:emphases1[0][0]].sum().long().item() if emphases1[0][0] > 0 else 0
        j1 = attn1[:emphases1[0][1]].sum().long().item()
        i2 = attn2[:emphases2[0][0]].sum().long().item() if emphases2[0][0] > 0 else 0
        j2 = attn2[:emphases2[0][1]].sum().long().item()
        
        # Step 1. Direct concatenation
        mu_y1_a, mu_y1_c = mu_y1[:, :, :i1], mu_y1[:, :, j1:y1_lengths]
        mu_y2_b = mu_y2[:, :, i2:j2]
        mu_y_edit = torch.cat((mu_y1_a, mu_y2_b, mu_y1_c), dim=2) 
        y_edit_lengths = int(mu_y_edit.shape[2])

        # Step 2. Soften junctions
        mu_y_edit = _soften_juntions(mu_y_edit, mu_y1, mu_y2, y_edit_lengths, y1_lengths, y2_lengths, i1, j1, i2, j2)

        y_edit_length_ = fix_len_compatibility_text_edit(y_edit_lengths)
        y_edit_lengths_tensor = torch.tensor([y_edit_lengths]).long().to(x1.device)
        y_edit_mask_for_scorenet = sequence_mask(y_edit_lengths_tensor, y_edit_length_).unsqueeze(1).to(mu_y1.dtype)
        
        eps1 = torch.randn_like(mu_y1, device=mu_y1.device) / temperature
        eps2 = torch.randn_like(mu_y2, device=mu_y1.device) / temperature
        eps_edit = torch.cat((eps1[:, :, :i1], eps2[:, :, i2:j2], eps1[:, :, j1:y1_lengths]), dim=2)
        z1 = mu_y1 + eps1
        z2 = mu_y2 + eps2
        z_edit = mu_y_edit + eps_edit

        if z_edit.shape[2] < y_edit_length_:
            pad = y_edit_length_ - z_edit.shape[2]
            zeros = torch.zeros_like(z_edit[:, :, :pad])
            z_edit = torch.cat((z_edit, zeros), dim=2)
            mu_y_edit = torch.cat((mu_y_edit, zeros), dim=2)
        elif z_edit.shape[2] > y_edit_length_:
            res = z_edit.shape[2] - y_edit_length_
            z_edit = z_edit[:, :, :-res]
            mu_y_edit = mu_y_edit[:, :, :-res]

        y_edit_mask_for_gradient = torch.zeros_like(mu_y_edit[:, :1, :])
        y_edit_mask_for_gradient[:, :, i1:i1+(j2-i2)] = 1

        dec1 = self.decoder(z1, y1_mask, mu_y1, n_timesteps, stoc)
        dec2, dec_edit = self.decoder.double_forward_text(z2, z_edit, mu_y2, mu_y_edit, y2_mask, y_edit_mask_for_scorenet, 
                                        y_edit_mask_for_gradient, i1, j1, i2, j2, n_timesteps, stoc, soften_mask, n_soften)

        dec1 = dec1[:, :, :y1_max_length]
        dec2 = dec2[:, :, :y2_max_length]
        dec_edit = dec_edit[:, :, :y_edit_lengths]
        dec_cat = torch.cat((dec1[:, :, :i1], dec2[:, :, i2:j2], dec1[:, :, j1:y1_lengths]), dim=2)

        return dec1, dec2, dec_edit, dec_cat