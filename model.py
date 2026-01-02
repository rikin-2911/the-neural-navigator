
import torch
import torch.nn as nn

VOCAB_SIZE = 10

class MultiModalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN Vision Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.img_proj = nn.Linear(64 * 32 * 32, 256)

        # Text Encoder
        self.text_emb = nn.Embedding(VOCAB_SIZE, 256)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        # Decoder → 10 × (x, y)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)
        )

    def forward(self, img, text):
        img_feat = self.cnn(img)
        img_token = self.img_proj(img_feat).unsqueeze(1)

        text_tokens = self.text_emb(text)

        fused = torch.cat([img_token, text_tokens], dim=1)
        fused = self.transformer(fused)

        out = self.decoder(fused[:, 0])
        return out.view(-1, 10, 2)