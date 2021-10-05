""" from https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS """

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'resources/filelists/train.txt'
valid_filelist_path = 'resources/filelists/valid.txt'
test_filelist_path = 'resources/filelists/test.txt'
cmudict_path = 'resources/cmu_dictionary'
n_feats = 80
add_blank = True

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = 'logs/new_exp'
test_size = 4
n_epochs = 10000
batch_size = 16
learning_rate = 1e-4
seed = 37
save_every = 1
out_size = fix_len_compatibility(2*22050//256)
