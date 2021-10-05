import argparse
import json
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence_for_editts, cmudict
from text.symbols import symbols
from utils import intersperse, intersperse_emphases

import sys, os

sys.path.append("./hifigan/")
from env import AttrDict
from models import Generator as HiFiGAN

torch.manual_seed(1234)

HIFIGAN_CONFIG = "./checkpts/hifigan-config.json"
HIFIGAN_CHECKPT = "./checkpts/hifigan.pt"
VOLUME_MAX = 32768
SAMPLE_RATE = 22050

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="path to a file with texts to synthesize",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="path to a checkpoint of Grad-TTS",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=1000,
        help="number of timesteps of reverse diffusion",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="out/content/wavs",
        help="directory path to save outuputs",
    )
    args = parser.parse_args()

    print("Initializing Grad-TTS...")
    generator = GradTTS(
        len(symbols) + 1,
        params.n_enc_channels,
        params.filter_channels,
        params.filter_channels_dp,
        params.n_heads,
        params.n_enc_layers,
        params.enc_kernel,
        params.enc_dropout,
        params.window_size,
        params.n_feats,
        params.dec_dim,
        params.beta_min,
        params.beta_max,
        params.pe_scale,
    )
    generator.load_state_dict(
        torch.load(args.checkpoint, map_location=lambda loc, storage: loc)
    )
    _ = generator.cuda().eval()
    print(f"Number of parameters: {generator.nparams}")

    print("Initializing HiFi-GAN...")
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(
        torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)["generator"]
    )
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    with open(args.file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict("./resources/cmu_dictionary")
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for i, text_list in enumerate(texts):
            print(f"[{i+1}/{len(texts)}] Synthesizing content-edited speech...")
            text1, text2 = text_list.split('#')
            sequence1, emphases1 = text_to_sequence_for_editts(text1, dictionary=cmu)     
            sequence2, emphases2 = text_to_sequence_for_editts(text2, dictionary=cmu)       
            x1 = torch.LongTensor(intersperse(sequence1, len(symbols))).cuda()[None]
            x2 = torch.LongTensor(intersperse(sequence2, len(symbols))).cuda()[None]
            emphases1 = intersperse_emphases(emphases1)
            emphases2 = intersperse_emphases(emphases2)
            x_lengths1 = torch.LongTensor([x1.shape[-1]]).cuda()
            x_lengths2 = torch.LongTensor([x2.shape[-1]]).cuda()

            y_dec1, y_dec2, y_dec_edit, y_dec_cat = generator.edit_content(
                x1,
                x2,
                x_lengths1,
                x_lengths2,
                emphases1,
                emphases2,
                n_timesteps=args.timesteps,
                temperature=1.5,
                stoc=False,
                length_scale=0.91,
            )

            audio1 = (
                vocoder(y_dec1).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)
            audio2 = (
                vocoder(y_dec2).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)
            audio_edit = (
                vocoder(y_dec_edit).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)
            audio_cat = (
                vocoder(y_dec_cat).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)

            write(os.path.join(args.save_dir, f"gen_{i}_gradtts-1.wav"), SAMPLE_RATE, audio1)
            write(os.path.join(args.save_dir, f"gen_{i}_gradtts-2.wav"), SAMPLE_RATE, audio2)
            write(os.path.join(args.save_dir, f"gen_{i}_EdiTTS.wav"), SAMPLE_RATE, audio_edit)
            write(os.path.join(args.save_dir, f"gen_{i}_baseline.wav"), SAMPLE_RATE, audio_cat)

    print(f"Check out {args.save_dir} folder for generated samples.")