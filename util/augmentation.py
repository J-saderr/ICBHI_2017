import random
import numpy as np
import nlpaug.augmenter.audio as naa

import torch
from torchvision.utils import _log_api_usage_once
from torchvision.transforms import transforms
from torchaudio import transforms as T

__all__ = ['augment_raw_audio']


def augment_raw_audio(sample, sample_rate, args):
    #print('augment_raw_audio')
    """
    Raw audio data augmentation technique
    you can utilize any library code
    1) nlpaug
    2) audiomentations
    3) librosa
    """

    """ 1) nlpaug """
    augment_list = [
        # naa.CropAug(sampling_rate=sample_rate)
        naa.NoiseAug(), # apply noise injection operation
        naa.SpeedAug(), # apply speed adjustment operation
        naa.LoudnessAug(factor=(0.5, 2)), # apply adjusting loudness operation
        naa.VtlpAug(sampling_rate=sample_rate, zone=(0.0, 1.0)), # apply vocal tract length perturbation (VTLP) operation
        naa.PitchAug(sampling_rate=sample_rate, factor=(-1,3)) # apply pitch adjustment operation
    ]

    # randomly sample augmentation
    aug_idx = random.randint(0, len(augment_list)-1)
    sample = augment_list[aug_idx].augment(sample)

    # apply all augmentations
    # for aug_idx in range(len(augment_list)):
    #     sample = augment_list[aug_idx].augment(sample)
    
    """ 2) audiomentations """
    # import audiomentations
    # from audiomentations import AddGaussianSNR, TimeStretch, PitchShift, Shift

    # # when using audiomentations library (not DEBUG yet)
    # audio_transforms = audiomentations.Compose([
    #     AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=40.0, p=0.5),
    #     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    #     PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    #     Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    # ])

    # sample = audio_transforms(samples=sample, sample_rate=sample_rate)

    """ 3) librosa """
    # import librosa

    # def _noise(data):
    #     noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    #     data = data + noise_amp * np.random.normal(size=data.shape[0])
    #     return data

    # def _stretch(data, rate=0.8):
    #     return librosa.effects.time_stretch(data, rate)

    # def _shift(data):
    #     shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    #     return np.roll(data, shift_range)
        
    # def _pitch(data, sampling_rate, pitch_factor=0.7):
    #     return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    # sample = _noise(sample)
    # sample = _stretch(sample)
    # sample = _shift(sample)
    # sample = _pitch(sample, sample_rate)

    if type(sample) == list:
        return sample[0]
    else:
        return sample