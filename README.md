# EdiTTS: Score-based Editing for Controllable Text-to-Speech

Official implementation of [EdiTTS: Score-based Editing for Controllable Text-to-Speech](https://arxiv.org/abs/2110.02584). Audio samples are available on our [demo page](https://editts.github.io).

## Abstract

> We present EdiTTS, an off-the-shelf speech editing methodology based on score-based generative modeling for text-to-speech synthesis. EdiTTS allows for targeted, granular editing of audio, both in terms of content and pitch, without the need for any additional training, task-specific optimization, or architectural modifications to the score-based model backbone. Specifically, we apply coarse yet deliberate perturbations in the Gaussian prior space to induce desired behavior from the diffusion model, while applying masks and softening kernels to ensure that iterative edits are applied only to the target region. Listening tests demonstrate that EdiTTS is capable of reliably generating natural-sounding audio that satisfies user-imposed requirements.

## Citation

Please cite this work as follows.

```bibtex
@inproceedings{tae22_interspeech,
  author={Jaesung Tae and Hyeongju Kim and Taesu Kim},
  title={{EdiTTS: Score-based Editing for Controllable Text-to-Speech}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={421--425},
  doi={10.21437/Interspeech.2022-6}
}
```

## Setup

1. Create a Python virtual environment (`venv` or `conda`) and install package requirements as specified in [`requirements.txt`](requirements.txt).

   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. Build the monotonic alignment module.

   ```sh
   cd model/monotonic_align
   python setup.py build_ext --inplace
   ```

For more information, refer to [the official repository of Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS).

## Checkpoints

The following checkpoints are already included as part of this repository, under [`checkpts`](checkpts). 

- [Grad-TTS (old ver.)](https://drive.google.com/drive/folders/1grsfccJbmEuSBGQExQKr3cVxNV0xEOZ7)
- [HiFi-GAN (LJ_FT_T2_V1 ver.)](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y)

## Pitch Shifting

1. Prepare an input file containing samples for speech generation. Mark the segment to be edited via a vertical bar separator, `|`. For instance, a single sample might look like

   > In | the face of impediments confessedly discouraging |

   We provide a sample input file in [`resources/filelists/edit_pitch_example.txt`](resources/filelists/edit_pitch_example.txt).

2. To run inference, type

   ```sh
   CUDA_VISIBLE_DEVICES=0 python edit_pitch.py \
       -f resources/filelists/edit_pitch_example.txt \
       -c checkpts/grad-tts-old.pt -t 1000 \
       -s out/pitch/wavs
   ```

   Adjust `CUDA_VISIBLE_DEVICES` as appropriate.

## Content Replacement

1. Prepare an input file containing pairs of sentences. Concatenate each pair with `#` and mark the parts to be replaced with a vertical bar separator. For instance, a single pair might look like

   > Three others subsequently | identified | Oswald from a photograph. #Three others subsequently | recognized | Oswald from a photograph.

   We provide a sample input file in [`resources/filelists/edit_content_example.txt`](resources/filelists/edit_content_example.txt).

2. To run inference, type

   ```sh
   CUDA_VISIBLE_DEVICES=0 python edit_content.py \
       -f resources/filelists/edit_content_example.txt \
       -c checkpts/grad-tts-old.pt -t 1000 \
       -s out/content/wavs
   ```

## References
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [SDEdit](https://github.com/ermongroup/SDEdit)

## License

Released under the [modified GNU General Public License](LICENSE).
