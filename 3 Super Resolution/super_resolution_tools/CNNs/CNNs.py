import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from ..error_quantifiers import multi_scale_ssim
import torch
from torch.utils.data import Dataset, DataLoader


# Convolutional block and residual block

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleResidual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        skip = self.skip(x)
        return self.relu(out + skip)



# Encoder

class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleResidual(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


# Decoder

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        x2 = f.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleResidual(in_ch, out_ch)

    def forward(self, x1, x2):
        return self.conv(self.up(x1, x2))


# Model

class ResUNet(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, lr=1e-3, lambda_ssim=0.2):

        # stored here for linting, otherwise available via self.hparams
        self.lr = lr
        self.lambda_ssim = lambda_ssim

        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.first_conv = DoubleConv(in_channels, 32)

        self.down1 = DownLayer(32, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256, 512)

        self.bottleneck = nn.Sequential(
            DoubleResidual(512, 512),
            nn.Dropout2d(p=0.1)
        )

        # Decoder
        self.up1 = UpLayer(512, 256)
        self.up2 = UpLayer(256, 128)
        self.up3 = UpLayer(128, 64)
        self.up4 = UpLayer(64, 32)

        self.last_conv = nn.Sequential(
            DoubleConv(32, 32),
            nn.Conv2d(32, out_channels, 1, bias=False)
        )

        self.criterion = nn.L1Loss()

        self.train_losses = []
        self.val_losses = []

    def forward(self, x):

        x1 = self.first_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.bottleneck(x5)

        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)

        return self.last_conv(x4_up)

    def compute_loss(self, outputs, labels):
        mse = self.criterion(outputs, labels)
        ssim_loss = 1 - multi_scale_ssim(outputs, labels)
        return mse + self.hparams.lambda_ssim * ssim_loss, mse, ssim_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss, mse, ssim_loss = self.compute_loss(out, y)
        self.train_losses.append(loss.item())

        self.log("train_loss", loss)
        self.log("train_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_ssim_loss", ssim_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss, mse, ssim_loss = self.compute_loss(out, y)
        self.val_losses.append(loss.item())

        self.log("val_loss", loss)
        self.log("val_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ssim_loss", ssim_loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


# Data loader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float().permute(0,3,1,2)
        self.y = y.float().permute(0,3,1,2)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataModule(pl.LightningDataModule):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=10, num_workers=7):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.X_train = torch.tensor(X_train)
        self.X_val   = torch.tensor(X_val)
        self.X_test  = torch.tensor(X_test)

        self.y_train = torch.tensor(y_train)
        self.y_val   = torch.tensor(y_val)
        self.y_test  = torch.tensor(y_test)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        self.train_ds = CustomDataset(self.X_train, self.y_train)
        self.val_ds   = CustomDataset(self.X_val, self.y_val)
        self.test_ds  = CustomDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)