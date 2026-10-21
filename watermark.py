import numpy as np 
import matplotlib.pyplot as plt 
import os 
import torch
import torchaudio
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Sampling(nn.Module):
    def __init__(self):
         super().__init__()
    def forward(self, mean, log_var):
         epsilon = torch.randn_like(mean)
         return mean + torch.exp(0.5 * log_var) *epsilon


class EnCoder(nn.Module):
    def __init__(self, input_channels = 1, num_classes = 1):
        super(EnCoder, self).__init__()
        channels = [input_channels, 16, 32, 64, 128, 256, 512]
        layers =[]
        for i in range (6):
            layers.append(
                       nn.Conv1d(in_channels=channels[i], out_channels = channels[i+1], kernel_size = 3, padding=1 ))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.feature_extractor = nn.Sequential(*layers)
        self.fc_mean = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1)
        self.fc_log_var = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1)
                       
    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_var = self.fc_log_var(features)
        return mean, log_var 
        

class Decoder(nn.Module):
    def __init__(self, output_channels = 1, num_classes = 1):
        super(Decoder, self).__init__()
        channels = [512, 256, 128, 64, 32, 16, output_channels]
        layers =[]
        for i in range (6):
            layers.append(
                       nn.ConvTranspose1d(in_channels=channels[i], out_channels = channels[i+1], kernel_size = 3, stride= 2, padding=1, output_padding=1))

            if i <5:
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
                       
    def forward(self, x):
        return self.network(x)




class Embedder(nn.Module):
    def __init__(self, latent_channels=512, watermark_size=256):
        super(Embedder, self).__init__()
        self.fc_watermark = nn.Linear(watermark_size, latent_channels)

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=latent_channels * 2, out_channels=latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=latent_channels, out_channels=latent_channels, kernel_size=3, padding=1)
        )

    def forward(self, z, watermark):
        w = self.fc_watermark(watermark)
        w = w.unsqueeze(2).expand_as(z)
        combined = torch.cat([z, w], dim=1)
        return self.network(combined)
        


class Detector(nn.Module):
    def __init__(self, input_channels=1, watermark_size=256, audio_length = 16000):
        super(Detector, self).__init__()
        self.network = nn.Sequential(nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
                                      nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                                      nn.LeakyReLU(0.2), nn.Flatten(), nn.Linear(32 * (audio_length//4), watermark_size),
                                      nn.Sigmoid())


    def forward(self, x):
        return self.network(x)


class AudioDataset(Dataset):
    def __init__(self, audio_dir, target_length=16000):
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
        self.target_length = target_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        current_length = waveform.shape[1]
        if current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            waveform = waveform[:, :self.target_length]
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

#In audio_dir you've to put directory where you have your samples
#dataset = AudioDataset(audio_dir="", target_length=16000)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = EnCoder().to(device)
decoder = Decoder().to(device)
embedder = Embedder().to(device)
detector = Detector().to(device)
sampler = Sampling().to(device)



optimizer_gen = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(embedder.parameters()), lr=1e-4)
optimizer_det = optim.Adam(detector.parameters(), lr=1e-4)

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

epochs = 50
watermark_size = 256
history_recon_loss = []
history_det_loss = []

for epoch in range(epochs):
    encoder.train()
    decoder.train()
    embedder.train()
    detector.train()
    
    for batch_idx, original_audio in enumerate(dataloader):
        original_audio = original_audio.to(device)
        batch_size = original_audio.size(0)
        watermark = torch.randint(0, 2, (batch_size, watermark_size), dtype=torch.float32).to(device)
        mean, log_var = encoder(original_audio)
        z = sampler(mean, log_var)
        z_watermarked = embedder(z, watermark)
        watermarked_audio = decoder(z_watermarked)
        optimizer_det.zero_grad()
        detected_watermark_det = detector(watermarked_audio.detach())
        det_loss = bce_loss(detected_watermark_det, watermark)
        
        det_loss.backward()
        optimizer_det.step()
        optimizer_gen.zero_grad()
        detected_watermark_gen = detector(watermarked_audio)
        det_loss_gen = bce_loss(detected_watermark_gen, watermark)       
        recon_loss = mse_loss(watermarked_audio, original_audio)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size
        total_gen_loss = 10.0 * recon_loss + 0.1 * kl_loss + 1.0 * det_loss_gen
        total_gen_loss.backward()
        optimizer_gen.step()

    history_recon_loss.append(recon_loss.item())
    history_det_loss.append(det_loss.item())

    print(f"Epoch [{epoch+1}/{epochs}] | Recon Loss: {recon_loss.item():.4f} | Det Loss: {det_loss.item():.4f}")
np.save('recon_loss_history.npy', np.array(history_recon_loss))
np.save('det_loss_history.npy', np.array(history_det_loss))
