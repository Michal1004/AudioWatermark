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
         return mean + torch.exp(0.5 * log_var) * epsilon

class EnCoder(nn.Module):
    def __init__(self, input_channels = 1):
        super(EnCoder, self).__init__()
        channels = [input_channels, 4, 8, 16, 32, 64, 128]
        layers =[]
        for i in range (6):
            layers.append(
                       nn.Conv1d(in_channels=channels[i], out_channels = channels[i+1], kernel_size = 3, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.feature_extractor = nn.Sequential(*layers)
        self.fc_mean = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.fc_log_var = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
                      
    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_var = self.fc_log_var(features)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, output_channels = 1):
        super(Decoder, self).__init__()
        channels = [128, 64, 32, 16, 8, 4, output_channels]
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
    def __init__(self, latent_channels=128, watermark_size=32):
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
    def __init__(self, input_channels=1, watermark_size=32, audio_length = 16000):
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
        waveform= torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform, sample_rate = torch.mean(waveform, dim=0, keepdim=True)
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

dataset = AudioDataset(audio_dir="/home/michal/Desktop/watermark/dataset/wavs", target_length=44100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
watermark_size = 32
epochs = 50


encoder = EnCoder().to(device)
decoder = Decoder().to(device)
embedder = Embedder().to(device)
detector = Detector().to(device)
sampler = Sampling().to(device)

# Jeden optymalizator dla generatora VAE+Embedder, drugi dla detektora
optimizer_gen = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(embedder.parameters()), lr=1e-4)
optimizer_det = optim.Adam(detector.parameters(), lr=1e-4)

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()


history_recon_loss = []
history_det_loss = []
start_epoch = 0

checkpoint_path = "watermark_checkpoint_WM32.pt"

if os.path.exists(checkpoint_path):
    print(f"Znaleziono plik {checkpoint_path}. Wczytywanie stanu modelu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    embedder.load_state_dict(checkpoint['embedder_state'])
    detector.load_state_dict(checkpoint['detector_state'])
    
    optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state'])
    optimizer_det.load_state_dict(checkpoint['optimizer_det_state'])
    
    start_epoch = checkpoint['epoch'] + 1
    history_recon_loss = checkpoint['history_recon_loss']
    history_det_loss = checkpoint['history_det_loss']
    
    print(f"Wczytano plik. Wznowienie uczenia od epoki {start_epoch + 1}.")
else:
    print("Nie znaleziono zapisanego punktu kontrolnego.")

for epoch in range(start_epoch, epochs):
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

    print(f"Podsumowanie epoki {epoch+1} | Recon Loss: {recon_loss.item():.6f} | Det Loss: {det_loss.item():.6f}")

    checkpoint = {
        'epoch': epoch,
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'embedder_state': embedder.state_dict(),
        'detector_state': detector.state_dict(),
        'optimizer_gen_state': optimizer_gen.state_dict(),
        'optimizer_det_state': optimizer_det.state_dict(),
        'history_recon_loss': history_recon_loss,
        'history_det_loss': history_det_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Stan modelu został zapisany.")


np.save('recon_loss_historyWM16_2.npy', np.array(history_recon_loss))
np.save('det_loss_historyWM16_2.npy', np.array(history_det_loss))
'''
# Upewnij się, że modele są w trybie ewaluacji (wyłącza to m.in. dropout czy batchnorm, jeśli by były)
encoder.eval()
embedder.eval()
decoder.eval()
detector.eval()

# Przykładowe wejścia (muszą mieć taki sam kształt jak dane treningowe)
# Batch size ustawiamy na 1 dla eksportu, ale dodamy "dynamic_axes",
# aby model ONNX przyjmował potem dowolny batch size.
dummy_audio = torch.randn(1, 1, 16000, device=device)
dummy_watermark = torch.randn(1, 32, device=device)

# ==========================================
# 1. Eksport Detektora
# ==========================================
print("Eksportowanie Detektora do ONNX...")
torch.onnx.export(
    detector,                                # Model do eksportu
    dummy_audio,                             # Przykładowe wejście
    "detector.onnx",                         # Nazwa pliku wyjściowego
    export_params=True,                      # Zapisz wagi modelu
    opset_version=11,                        # Wersja ONNX (11 lub 14 są najbardziej stabilne)
    do_constant_folding=True,                # Optymalizacja modelu
    input_names=['audio_input'],             # Nazwa wejścia
    output_names=['watermark_output'],       # Nazwa wyjścia
    dynamic_axes={                           # Pozwala na dynamiczny batch size w ONNX
        'audio_input': {0: 'batch_size'},
        'watermark_output': {0: 'batch_size'}
    }
)
print("Zapisano: detector.onnx")

# ==========================================
# 2. Eksport Generatora (Nakładania znaku)
# ==========================================
# Tworzymy klasę-wrapper, aby połączyć 3 modele w jeden potok (pipeline)
class WatermarkInference(nn.Module):
    def __init__(self, encoder, embedder, decoder):
        super().__init__()
        self.encoder = encoder
        self.embedder = embedder
        self.decoder = decoder

    def forward(self, audio, watermark):
        # Pomijamy Sampling (torch.randn_like) - w inferencji VAE używa się tylko mean
        mean, _ = self.encoder(audio)
        z_watermarked = self.embedder(mean, watermark)
        return self.decoder(z_watermarked)

# Inicjalizujemy wrapper naszymi wytrenowanymi modelami
generator = WatermarkInference(encoder, embedder, decoder).to(device)
generator.eval()

print("Eksportowanie Generatora do ONNX...")
torch.onnx.export(
    generator,
    (dummy_audio, dummy_watermark),          # Krotka dla wielu wejść
    "generator.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['audio_input', 'watermark_input'],
    output_names=['audio_output'],
    dynamic_axes={
        'audio_input': {0: 'batch_size'},
        'watermark_input': {0: 'batch_size'},
        'audio_output': {0: 'batch_size'}
    }
)
print("Zapisano: generator.onnx")
'''
