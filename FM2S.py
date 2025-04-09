import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage as nd
import torch.nn.functional as F
from einops import rearrange

SEED = 3407

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_num = 450
max_epoch = 5

if device == torch.device("cpu"):
    print("[WARNING] You are running on CPU instead of CUDA!")

class network(nn.Module):
    def __init__(self, n_chan):
        super(network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=1e-3)
        self.conv1 = nn.Conv2d(n_chan, 24, 5, padding=2)
        self.conv2 = nn.Conv2d(24, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, n_chan, 5, padding=2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


def noise_addition(img, config):
    B, C, H, W = img.shape
    stride = config['stride']
    new_H = ((H + stride - 1) // stride) * stride
    new_W = ((W + stride - 1) // stride) * stride
    pad_h = new_H - H
    pad_w = new_W - W
    
    img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
    patches = img_padded.unfold(2, stride, stride).unfold(3, stride, stride)
    noise_idx = patches.mean(dim=(1, 4, 5)).clamp(1e-5, 0.15)
    gaussian_level = config['g_map'] * noise_idx
    poisson_level  = config['p_map'] / noise_idx
    gaussian_level_exp = rearrange(gaussian_level, 'b n_h n_w -> b 1 n_h n_w 1 1')
    poisson_level_exp = rearrange(poisson_level, 'b n_h n_w -> b 1 n_h n_w 1 1')
    patches_noisy = torch.poisson(patches * poisson_level_exp) / poisson_level_exp
    gaussian_noise = torch.normal(mean=torch.zeros_like(patches_noisy),
                                  std=(gaussian_level_exp / 255))
    patches_noisy = patches_noisy + gaussian_noise
    patches_noisy = torch.clamp(patches_noisy, 0, 1)
    noisy_img_padded = rearrange(patches_noisy, 'b c n_h n_w new_h new_w -> b c (n_h new_h) (n_w new_w)')
    noisy_img = noisy_img_padded[:, :, :H, :W]
    noisy_img = torch.poisson(noisy_img * config['lam_p']) / config['lam_p']
    noisy_img = torch.clamp(noisy_img, 0, 1)
    return noisy_img

def FM2S(raw_noisy_img, config:dict):
    raw_noisy_img = raw_noisy_img / 255
    clean_img = nd.median_filter(raw_noisy_img, config['w_size'])
    clean_img = torch.tensor(clean_img, dtype=torch.float32, device=device)
    clean_img = clean_img.unsqueeze(0).repeat(1, config['n_chan'], 1, 1)
    raw_noisy_img = torch.tensor(raw_noisy_img, dtype=torch.float32, device=device)
    raw_noisy_img = raw_noisy_img.unsqueeze(0).repeat(1, config['n_chan'], 1, 1)

    model = network(config['n_chan']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for i in range(5):
        pred = model(raw_noisy_img)
        loss = criterion(pred, clean_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    for i in range(train_num):
        noisy_img = noise_addition(clean_img, config)
        for _ in range(max_epoch):
            pred = model(noisy_img)
            loss = criterion(pred, clean_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        denoised_img = model(raw_noisy_img)

    denoised_img = torch.clamp(denoised_img, 0, 1) * 255
    denoised_img = torch.mean(denoised_img, dim=1).squeeze()
    denoised_img = denoised_img.cpu().int().numpy()
    return denoised_img
