from torch import nn
from torchaudio.models.conformer import Conformer


class DepthwiseConv(nn.Module):
    """
    Depthwise convolution module
    """

    def __init__(self, conv_channels: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=conv_channels,
            ),
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=1,
                stride=1,
                groups=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseConvSubsampling(nn.Module):
    """
    DepthwiseConvSubsampling subsampling module from torchaudio.
    """

    def __init__(
        self,
        n_feats: int,
        d_model: int,
        subsampling_factor: int = 4,
        conv_channels: int = 256,
    ):
        super().__init__()

        if subsampling_factor not in [4, 8]:
            raise ValueError("subsampling_factor must be 4 or 8")

        layers = [
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        ]
        self.num_subsampling_layers = {4: 2, 8: 3}[subsampling_factor]

        for _ in range(self.num_subsampling_layers - 1):
            layers.append(DepthwiseConv(conv_channels=conv_channels))

        self.conv = nn.Sequential(*layers)

        out_freq = n_feats
        for _ in range(self.num_subsampling_layers):
            out_freq = (out_freq + 1) // 2

        out_dim = conv_channels * out_freq
        self.out = nn.Linear(out_dim, d_model)

    def recalculate_output_lengths(self, input_lengths):
        output_lengths = input_lengths
        for _ in range(self.num_subsampling_layers):
            output_lengths = (output_lengths + 1) // 2
        return output_lengths

    def forward(self, x, x_lengths):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T, C * F)
        x = self.out(x)
        x_lengths = self.recalculate_output_lengths(x_lengths)
        return x, x_lengths


class FastConformerModel(nn.Module):
    """
    FastConformer model
    """

    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        subsampling_factor: int = 4,
        conv_channels: int = 256,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.subsampling = DepthwiseConvSubsampling(
            n_feats=n_feats,
            d_model=input_dim,
            subsampling_factor=subsampling_factor,
            conv_channels=conv_channels,
        )

        self.drop = nn.Dropout(p=dropout)

        self.net = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )

        self.output_layer = nn.Linear(in_features=input_dim, out_features=n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram, spectrogram_length = self.subsampling(
            spectrogram.transpose(1, 2), spectrogram_length
        )
        spectrogram = self.drop(spectrogram)
        features, output_lengths = self.net(
            spectrogram, spectrogram_length.to(spectrogram.device)
        )
        features = self.output_layer(features)
        log_probs = nn.functional.log_softmax(features, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": output_lengths.cpu()}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
