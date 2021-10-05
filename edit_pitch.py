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
        default="out/pitch/wavs",
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
        for i, text in enumerate(texts):
            print(f"[{i+1}/{len(texts)}] Synthesizing pitch-edited speech...")
            sequence, emphases = text_to_sequence_for_editts(text, dictionary=cmu)       
            x = torch.LongTensor(intersperse(sequence, len(symbols))).cuda()[None]
            emphases = intersperse_emphases(emphases)
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

            y_dec, y_baseline_up, y_edit_up = generator.edit_pitch(
                x,
                x_lengths,
                n_timesteps=args.timesteps,
                temperature=1.5,
                stoc=False,
                length_scale=0.91,
                emphases=emphases,
                direction='up'
            )
            _, y_baseline_down, y_edit_down = generator.edit_pitch(
                x,
                x_lengths,
                n_timesteps=args.timesteps,
                temperature=1.5,
                stoc=False,
                length_scale=0.91,
                emphases=emphases,
                direction='down'
            )

            audio_gradtts = (
                vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)
            audio_baseline_up = (
                vocoder(y_baseline_up).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)
            audio_baseline_down = (
                vocoder(y_baseline_down).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)
            audio_edit_up = (
                vocoder(y_edit_up).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)
            audio_edit_down = (
                vocoder(y_edit_down).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
            ).astype(np.int16)

            write(os.path.join(save_dir, f'gen_{i+1}_gradtts.wav'), SAMPLE_RATE, audio_gradtts)
            write(os.path.join(save_dir, f'gen_{i+1}_baseline-up.wav'), SAMPLE_RATE, audio_baseline_up)
            write(os.path.join(save_dir, f'gen_{i+1}_baseline-down.wav'), SAMPLE_RATE, audio_baseline_down)
            write(os.path.join(save_dir, f'gen_{i+1}_EdiTTS-up.wav'), SAMPLE_RATE, audio_edit_up)
            write(os.path.join(save_dir, f'gen_{i+1}_EdiTTS-down.wav'), SAMPLE_RATE, audio_edit_down)

    print(f"Check out {args.save_dir} folder for generated samples.")
