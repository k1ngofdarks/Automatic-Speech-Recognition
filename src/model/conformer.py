from torch import nn
from torchaudio.models.conformer import Conformer


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling module from torchaudio.
    """

    def __init__(self, n_feats: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        out_freq = (n_feats - 3) // 2 + 1
        out_freq = (out_freq - 3) // 2 + 1
        out_dim = d_model * out_freq
        self.out = nn.Linear(out_dim, d_model)

    def recalculate_output_lengths(self, input_lengths):
        output_lengths = (input_lengths - 3) // 2 + 1
        output_lengths = (output_lengths - 3) // 2 + 1
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


class ConformerModel(nn.Module):
    """
    Conformer model from torchaudio.
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
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.subsampling = ConvSubsampling(n_feats=n_feats, d_model=input_dim)

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
