import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
import torchaudio
# from scipy.io.wavfile import read

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
# from tacotron2.layers import TacotronSTFT
from .preparation import data_prepar_audio

MAX_WAV_VALUE = 32768

# def files_to_list(data_path):
#     """
#     Load all .wav files in data_path
#     """
#     files = [os.path.join(data_path, f.rstrip()) for f in os.listdir(data_path) if len(f)>=4 and f[-4:]=='.wav']
#     return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = torchaudio.load(full_path)
    return data.float(), sampling_rate


class Dataset_Audio(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_file, valid): 
        
        self.valid = valid
        self.segment_length = MAX_WAV_VALUE
        
        self.audio_files, self.org_audio_files = data_prepar_audio(data_file)   
        random.seed(1234)
        random.shuffle(self.audio_files)  
        random.seed(1234)
        random.shuffle(self.org_audio_files)  
       
    
    def cut_and_pad(self, audio):
        if self.valid == True:
            # whole audio for valid set
            pass
        
        else: # training
            if audio.size(1) >= self.segment_length:
                max_audio_start = audio.size(1) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start + self.segment_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(1)), 'constant').data
                
        return audio
    
        
    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        org_audio_filename = self.org_audio_files[index]
        
        audio, sampling_rate = load_wav_to_torch(filename)
        audio_org, sampling_rate_org = load_wav_to_torch(org_audio_filename)
        # Take segment
        audio = self.cut_and_pad(audio)
        audio_org = self.cut_and_pad(audio_org)        

        return (audio, audio_org)
            
        
    def __len__(self):
        return len(self.audio_files)

    
    
# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
# if __name__ == "__main__":
#     # Get defaults so it can work with no Sacred
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', "--filelist_path", required=True)
#     parser.add_argument('-c', '--config', type=str,
#                         help='JSON file for configuration')
#     parser.add_argument('-o', '--output_dir', type=str,
#                         help='Output directory')
#     args = parser.parse_args()

#     with open(args.config) as f:
#         data = f.read()
#     data_config = json.loads(data)["validset_config"]
#     mel2samp = Mel2Samp(**data_config)

#     filepaths = files_to_list(args.filelist_path)

#     # Make directory if it doesn't exist
#     if not os.path.isdir(args.output_dir):
#         os.makedirs(args.output_dir)
#         os.chmod(args.output_dir, 0o775)

#     for filepath in filepaths:
#         audio, sr = load_wav_to_torch(filepath)
#         melspectrogram = mel2samp.get_mel(audio)
#         filename = os.path.basename(filepath)
#         new_filepath = args.output_dir + '/' + filename + '.pt'
#         print(new_filepath)
#         torch.save(melspectrogram, new_filepath)

        