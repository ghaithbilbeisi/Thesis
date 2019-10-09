"""Image Encoder."""
import torch.nn as nn
import torch.nn.functional as F
import torch

from onmt.encoders.encoder import EncoderBase


class ImageEncoder(EncoderBase):
    """A simple encoder CNN -> RNN for image src.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout,
                 image_chanel_size=3):
        super(ImageEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        ## Block 1
        self.layer1 = nn.Conv2d(image_chanel_size, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        ## Block 2
        self.layer3 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(128, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(128, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(128, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        ## Block 3
        self.layer7 = nn.Conv2d(128, 128, kernel_size=(2, 2),
                                padding=(1, 1), stride=(1, 1))
        self.layer8 = nn.Conv2d(128, 128, kernel_size=(2, 2),
                                padding=(1, 1), stride=(1, 1))
        self.layer9 = nn.Conv2d(128, 128, kernel_size=(2, 2),
                                padding=(1, 1), stride=(1, 1))
        self.layer10 = nn.Conv2d(128, 128, kernel_size=(2, 2),
                                padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(128)

        src_size = 128
        dropout = dropout[0] if type(dropout) is list else dropout
        self.rnn = nn.LSTM(src_size, int(rnn_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.pos_lut = nn.Embedding(1000, src_size)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with ImageEncoder.")
        # why is the model_opt.__dict__ check necessary?
        if "image_channel_size" not in opt.__dict__:
            image_channel_size = 3
        else:
            image_channel_size = opt.image_channel_size
        return cls(
            opt.enc_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            image_channel_size
        )

    def load_pretrained_vectors(self, opt):
        """Pass in needed options only when modify function definition."""
        pass

    def forward(self, src, lengths=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""

        batch_size = src.size(0)
        # (batch_size, 64, imgH, imgW)
        # Block 1
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)
        src = F.relu(self.layer2(src), True)
        # max pool 1 (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))
        # batch norm 1
        src = self.batch_norm1(src)

        # (batch_size, 64, imgH/2, imgW/2)
        # Block 2
        src = F.relu(self.layer3(src), True)
        src = F.relu(self.layer4(src), True)
        src = F.relu(self.layer5(src), True)
        src = F.relu(self.layer6(src), True)
        # max pool 2 (batch_size, 128, imgH/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))
        # batch norm 2 (batch_size, 128, imgH/2, imgW/2/2)
        src = self.batch_norm2(src)

        # (batch_size, 128, imgH/2, imgW/2/2)
        # Block 3
        src = F.relu(self.layer7(src), True)
        src = F.relu(self.layer8(src), True)
        src = F.relu(self.layer9(src), True)
        src = F.relu(self.layer10(src), True)
        # max pool 3 (batch_size, 128, imgH/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(1,2), stride=(1,2))
        # batch norm 3 (batch_size, 128, imgH/2/2, imgW/2/2)
        src = self.batch_norm3(src)

        # # (batch_size, 128, H/2/2, W/2/2)
        all_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data) \
                .long().fill_(row)
            pos_emb = self.pos_lut(row_vec)
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return hidden_t, out, lengths

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
