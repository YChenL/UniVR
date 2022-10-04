import torch
from torch.utils.data.distributed import DistributedSampler
from .mel2samp import Mel2Samp
from .audio_dataset import Dataset_Audio


# def collate_fn(data):
#     for unit in data:
#         batch_sequence = list(map(lambda x: torch.tensor(x[findex]), x_data))
#         batch_data[feat] = torch.nn.utils.rnn.pad_sequence(batch_sequence).T

#     return {x: torch.tensor(unit_x),  y: torch.tensor(unit_y), }



def load_LJSpeech(data_file, valid=False, segment_length=16000, filter_length=1024, n_mel_channels=128,
                  hop_length=256, win_length=1024, mel_fmin=0, mel_fmax=8000,
                  batch_size=32, num_gpus=1, target_length=500, speaker_emb=False, joint_factor=False): #sampling_rate=22050,
    
    LJSpeech_dataset = Mel2Samp(data_file=data_file, valid=valid, segment_length=segment_length, filter_length=filter_length,
                                n_mel_channels=n_mel_channels, hop_length=hop_length, win_length=win_length,
                                mel_fmin=mel_fmin, mel_fmax=mel_fmax, target_length=target_length, 
                                speaker_emb=speaker_emb, joint_factor=joint_factor) #sampling_rate=sampling_rate,

    # distributed sampler
    train_sampler = DistributedSampler(LJSpeech_dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(LJSpeech_dataset, 
                                              batch_size=batch_size,  
                                              sampler=train_sampler,
                                              num_workers=4,
                                              pin_memory=False,
                                              drop_last=True,
                                              ) #collate_fn=collate_fn
    return trainloader


def load_Audiodata(data_file, valid=False, batch_size=16, num_gpus=1): 
    Audio_dataset = Dataset_Audio(data_file=data_file, valid=valid) 

    # distributed sampler
    train_sampler = DistributedSampler(Audio_dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(Audio_dataset, 
                                              batch_size=batch_size,  
                                              sampler=train_sampler,
                                              num_workers=4,
                                              pin_memory=False,
                                              drop_last=True,
                                              ) #collate_fn=collate_fn
    return trainloader

