# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax, FairseqDropout
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model("lstm_sa")
class LSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--windows_size', type=int, metavar='N',
                            help='windows size of local attention')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError("--encoder-layers must match --decoder-layers")

        max_source_positions = getattr(
            args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )
        max_target_positions = getattr(
            args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError("--share-all-embeddings requires a joint dictionary")
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embed not compatible with --decoder-embed-path"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to "
                    "match --decoder-embed-dim"
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim,
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim
        ):
            raise ValueError(
                "--share-decoder-input-output-embeddings requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=utils.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            residuals=False,
            windows_size=args.windows_size,
        )
        return cls(encoder, decoder)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        # print("传入模型")
        # print(src_tokens)
        # print(src_lengths)
        # print(prev_output_tokens) [2,34]

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        # print(encoder_out)

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )  # yt-1,

        return decoder_out


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
            self,
            dictionary,
            embed_dim=512,
            hidden_size=512,
            num_layers=1,
            dropout_in=0.1,
            dropout_out=0.1,
            bidirectional=False,
            left_pad=True,
            pretrained_embed=None,
            padding_idx=None,
            max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
            window_size=7,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        # print(num_embeddings)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2
        self.window_size = window_size

    def forward(
            self,
            src_tokens: Tensor,
            src_lengths: Tensor,
            enforce_sorted: bool = True,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        src_lengths = src_lengths + (2 * self.window_size + 1)
        # print(src_tokens.shape) # 2 x 33 33为batch中最大长度
        # print(src_lengths) # 33, 15
        # tensor([[3, 6, 4, 3, 3, 3, 8, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3,
        #          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 12, 5, 2],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3,
        #          8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]],
        #        )

        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )
        src_tokens = self.pad_with_window_size(src_tokens)

        # print(src_tokens)

        # print(type(src_tokens))
        # print(src_tokens.shape) torch.Size([10, 67]) 10个句子
        # [6, 4, 14, 86, 276, 271, 46, 5, 124, 181, 241, 8, 156, 28,
        #  220, 153, 265, 212, 4, 30, 9, 72, 20, 125, 5, 52, 16, 213,
        #  245, 131, 282, 151, 232, 49, 2, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

        bsz, seqlen = src_tokens.size()  # bsz 2, seqlen 33
        # print(type(bsz))
        # print(bsz)
        # print(type(seqlen))
        # print(seqlen)

        # embed tokens 如果有与训练的emddding
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C  seqlen x bsz x emd_dim 适应nn.lstm
        x = x.transpose(0, 1)  # 33, 2, 16

        # print(x)
        # print(x.shape)

        # pack embedded source tokens into a PackedSequence 压缩填充的pad
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.cpu(), enforce_sorted=enforce_sorted
        )

        # print(packed_x[1].shape) 33
        # print(packed_x[0].shape) # 48, 16
        # print(packed_x[1])
        # tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1]) 前15个填满batch 2 ，后面18个只有一个句子为1  15 x 2 + 18 =48

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        # print(state_size)  # (1, 2, 16)
        h0 = x.new_zeros(*state_size)  # (1, 2, 16) 大小，初始隐藏层状态 全都是0
        c0 = x.new_zeros(*state_size)  # (1, 2, 16) 大小，初始细胞状态 全都是0
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))
        # print(packed_x[1].shape) # 33
        # print(packed_x[0].shape) # 48, 16
        # print(final_hiddens) # torch.Size([1, 2, 16])
        # print(final_cells) # torch.Size([1, 2, 16])

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_idx * 1.0
        )
        x = self.dropout_out_module(x)
        # print(x)  # 33, 2, 16

        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        # print(encoder_padding_mask.shape)  # 33, 2
        # tensor([[False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, False],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True],
        #         [False, True]])

        # print(x)

        return tuple(
            (
                x,  # seq_len x batch x hidden [33, 2, 16] 每个时间步的隐藏层状态
                final_hiddens,  # num_layers x batch x num_directions*hidden [1, 2, 16] 最后一步隐藏层状态
                final_cells,  # num_layers x batch x num_directions*hidden [1, 2, 16] 最后一步的细胞状态
                encoder_padding_mask,  # seq_len x batch 69x10 pad mask矩阵
            )
        )

    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
                encoder_out[3].index_select(1, new_order),
            )
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

    def pad_with_window_size(self, batch):
        # print("===========================")
        # print(batch.shape)  # [31, 518]
        batch = batch.transpose(0, 1)
        size = batch.size()
        # print(size)  #[31, 518] batch.shape
        n = len(size)
        if n == 2:
            length, batch_size = size
            padded_length = length + (2 * self.window_size + 1)
            padded = torch.empty((padded_length, batch_size), dtype=torch.long).cuda()
            padded[:self.window_size, :] = self.padding_idx
            padded[self.window_size:self.window_size + length, :] = batch
            padded[-(self.window_size + 1):, :] = self.padding_idx
        # elif n == 3:
        #     length, batch_size, hidden = size
        #     padded_length = length + (2 * self.window_size + 1)
        #     padded = torch.empty((padded_length, batch_size, hidden), dtype=torch.long).cuda()
        #     padded[:self.window_size, :, :] = self.padding_idx
        #     padded[self.window_size:self.window_size + length, :, :] = batch
        #     padded[-(self.window_size + 1):, :, :] = self.padding_idx
        else:
            raise Exception(f'Cannot pad batch with {n} dimensions.')
        return padded.transpose(0, 1)


"""
class Attention(nn.Module):

    def __init__(self, window_size, hidden_size, device):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.std_squared = (self.window_size / 2) ** 2
        self.hidden_size = hidden_size
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=math.ceil(hidden_size / 2))
        self.fc2 = nn.Linear(in_features=math.ceil(hidden_size / 2), out_features=1)

    def forward(self, encoder_output, decoder_output, lengths, T, batch_size, output_weights):
        s0 = lengths[0].item()
        lengths = lengths.view(batch_size, 1)
        window_length = 2 * self.window_size + 1
        # h_s: batch_size x (window_size + S + window_size) x hidden
        h_s = encoder_output
        h_s = h_s.permute(1, 0, 2)
        # h_t: batch_size x T x hidden
        h_t = decoder_output
        h_t = h_t.permute(1, 0, 2)

        # batch_size x T x 1
        p = self.tanh(self.fc1(h_t))
        p = self.sigmoid(self.fc2(p))
        p = p.view(batch_size, T)
        p = self.window_size + lengths.float() * p
        p = p.unsqueeze(2)

        window_start = torch.round(p - self.window_size).int()
        window_end = window_start + window_length
        positions = torch.empty((batch_size, T, window_length), device=self.device, dtype=torch.float)
        selection = torch.empty((batch_size, window_length, self.hidden_size), device=self.device, dtype=torch.float)
        for i in range(batch_size):
            for j in range(T):
                start = window_start[i, j].item()
                end = window_end[i, j].item()
                positions[i, j] = torch.arange(start, end, device=self.device, dtype=torch.float)
                selection[i] = h_s[i, start:end]

        # batch_size x T x window_length
        gaussian = torch.exp(-(positions - p) ** 2 / (2 * self.std_squared))
        gaussian = gaussian.view(batch_size, T, window_length)

        # batch_size x T x window_length
        epsilon = 1e-14
        score = self.score(selection, h_t)
        for i in range(batch_size):
            li = lengths[i].item()
            for j in range(T):
                start = window_start[i, j].item()
                end = window_end[i, j].item()
                if start < self.window_size:
                    d = self.window_size - start
                    score[i, j, :d] = epsilon
                if end > li + self.window_size:
                    d = (li + self.window_size) - end
                    score[i, j, d:] = epsilon

        # batch_size x T x window_length
        a = self.softmax(score)
        a = a * gaussian

        # batch_size x T x hidden_size
        c = torch.bmm(a, selection)

        # T x batch_size x hidden_size
        c = c.permute(1, 0, 2)

        if not output_weights:
            return c

        # insert weights of first sentence for eventual visualiation
        weights = torch.zeros((T, s0), device=self.device, dtype=torch.float)
        for j in range(T):
            start = window_start[0, j].item()
            end = window_end[0, j].item()
            if start < self.window_size and end > self.window_size + s0:
                # overflow in both ends
                weights_start = 0
                weights_end = s0
                a_start = self.window_size - start
                a_end = a_start + s0
            elif start < self.window_size:
                # overflow in left side only
                weights_start = 0
                weights_end = end - self.window_size
                a_start = self.window_size - start
                a_end = window_length
            elif end > self.window_size + s0:
                # overflow in right side only
                weights_start = start - self.window_size
                weights_end = s0
                a_start = 0
                a_end = a_start + (weights_end - weights_start)
            else:
                # a is contained in sentence
                weights_start = start - self.window_size
                weights_end = end - self.window_size
                a_start = 0
                a_end = window_length
            weights[j, weights_start:weights_end] = a[0, j, a_start:a_end]

        return c, weights

    def score(self, h_s, h_t):
        # h_s : batch x length x hidden
        # h_t : batch x T x hidden
        h_s = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s)
"""


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, window_size, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )
        self.window_size = window_size
        self.fc1 = nn.Linear(in_features=input_embed_dim, out_features=math.ceil(input_embed_dim / 2))
        self.fc2 = nn.Linear(in_features=math.ceil(input_embed_dim / 2), out_features=1)
        self.std_squared = (self.window_size / 2) ** 2
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim, [2, 16]
        # source_hids: srclen x bsz x source_embed_dim, [33, 2, 16] encoder每个时间步的隐藏层状态
        # source_hids: ("hs"), encoder 的隐藏层输出，每次调用的都是一样的
        current_batch_size = input.size()[0]
        window_length = 2 * self.window_size + 1

        len_padding_mask = torch.where(encoder_padding_mask == False, 1, 0)
        # print(encoder_padding_mask)
        lengths = torch.sum(len_padding_mask, dim=0).cuda()  # [2] [33, 15]
        lengths = lengths.view(current_batch_size, 1)
        # print(encoder_padding_mask.shape)

        # x: bsz x source_embed_dim ([2, 16]) 每解码一个token计算一次 attention score
        x = self.input_proj(input)  # 当前时间步 decoder 的隐藏层状态 current target hidden state "ht" 34(目标句子长度)次
        x = x.unsqueeze(2)  # 2,16,1
        x = x.permute(0, 2, 1)

        source_hids = source_hids.permute(1, 0, 2)  # 2，7，16

        p = self.tanh(self.fc1(x))
        p = self.sigmoid(self.fc2(p))
        p = p.view(current_batch_size, 1)
        p = self.window_size + lengths.float() * p
        p = p.unsqueeze(2)

        # source_hids: srclen x bsz x source_embed_dim, [33, 2, 16] encoder每个时间步的隐藏层状态
        selection = torch.empty((self.window_size + 1, input.size()[0],
                                 input.size()[1]))  # window_size x bsz x source_embed_dim  5,2,16
        window_start = torch.round(p - self.window_size).int()
        window_end = window_start + window_length
        positions = torch.empty((current_batch_size, 1, window_length), dtype=torch.float).cuda()
        selection = torch.empty((current_batch_size, window_length, input.size()[1]), dtype=torch.float).cuda()
        for i in range(current_batch_size):
            start = window_start[i, 0].item()
            end = window_end[i, 0].item()
            positions[i, 0] = torch.arange(start, end, dtype=torch.float).cuda()
            selection[i] = source_hids[i, start:end]

        # batch_size x T x window_length
        gaussian = torch.exp(-(positions - p) ** 2 / (2 * self.std_squared))
        gaussian = gaussian.view(current_batch_size, 1, window_length)

        # batch_size x T x window_length
        epsilon = 1e-14
        score = self.score(selection, x)
        for i in range(current_batch_size):
            li = lengths[i].item()
            start = window_start[i, 0].item()
            end = window_end[i, 0].item()
            if start < self.window_size:
                d = self.window_size - start
                score[i, 0, :d] = epsilon
            if end > li + self.window_size:
                d = (li + self.window_size) - end
                score[i, 0, d:] = epsilon

        # batch_size x T x window_length
        a = self.softmax(score)
        a = a * gaussian  # [4, 1, 15]

        scores = a.squeeze(dim=1)

        # batch_size x T x hidden_size
        c = torch.bmm(a, selection)
        # T x batch_size x hidden_size
        c = c.permute(1, 0, 2)
        c = c.squeeze(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((c, input), dim=1)))  # [2, 16] attention vector ht-

        return x, scores, window_start, window_end

    def score(self, h_s, h_t):
        # h_s : batch x length x hidden
        # h_t : batch x T x hidden
        h_s = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s)


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
            self,
            dictionary,
            embed_dim=512,
            hidden_size=512,
            out_embed_dim=512,
            num_layers=1,
            dropout_in=0.1,
            dropout_out=0.1,
            attention=True,
            encoder_output_units=512,
            pretrained_embed=None,
            share_input_output_embed=False,
            adaptive_softmax_cutoff=None,
            max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
            residuals=False,
            windows_size=7,
    ):
        super().__init__(dictionary)
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        # print(num_embeddings)
        padding_idx = dictionary.pad()  # 1
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units  # [33, 2, 16]
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    input_size=input_feed_size + embed_dim
                    if layer == 0
                    else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )
        # ModuleList(
        #     (0): LSTMCell(32, 16)
        # )

        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(
                hidden_size, encoder_output_units, hidden_size, windows_size, bias=False
            )  # 16, 16, 16
        else:
            self.attention = None

        # print(hidden_size) 16
        # print(out_embed_dim) 16

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                hidden_size,
                adaptive_softmax_cutoff,
                dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            src_lengths: Optional[Tensor] = None,
    ):
        """
        encoder_out:
                    x,  # seq_len x batch x hidden [33, 2, 16] 每个时间步的隐藏层状态
                    final_hiddens,  # num_layers x batch x num_directions*hidden [1, 2, 16] 最后一步隐藏层状态
                    final_cells,  # num_layers x batch x num_directions*hidden [1, 2, 16] 最后一步的细胞状态
                    encoder_padding_mask,  # seq_len x batch [33, 2] pad mask矩阵
        """

        # print(prev_output_tokens.shape) [2, 34]

        # print(incremental_state) None
        # print(src_lengths) None
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        # print(x.shape) ([2, 33, 16])
        # if attn_scores != None:
        #     print(attn_scores.shape)  # [2, 37, 33]
        return self.output_layer(x), attn_scores

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)

        # window_padding_mask = torch.full_like(encoder_padding_mask, True)
        # print(window_padding_mask)
        srclen = encoder_outs.size(0)  # 源语言句子长度（with pad）

        # print(prev_output_tokens.shape)  # [2, 34], 目标句子长度34

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]  # 取最后一个 token

        bsz, seqlen = prev_output_tokens.size()
        # print(bsz)  # 2
        # print(seqlen)  # 34 目标句子长度

        # embed tokens， 将目标句子转化为 embed, [2, 34] -> [2, 34, 16]
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)  # [34, 2, 16]

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert (
                srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores = (
            x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        )  # [33, 34, 2]  初始化为全0
        # print(attn_scores)

        outs = []
        # seqlen = 34 ，遍历目标句子
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                # print(x)  # [34, 2, 16]
                # print(input_feed)  # [2, 16]
                input = torch.cat((x[j, :, :], input_feed), dim=1)  # x[j, :, :] --> [2, 16] 在第二个维度上拼接
                # print(input)  # [2, 32]
            else:
                input = x[j]  # [2, 16], 如果不用input feed， 就只取当前token的第j维度的值，但默认使用 input feed

            for i, rnn in enumerate(self.layers):
                # print(i, rnn)  # 0 LSTMCell(32, 16)
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None  # [33, 34, 2], 第一次全部为0
                # print(hidden.shape)  # [2, 16] decoder当前时间步的隐藏层状态
                # print(encoder_outs.shape)  # [33, 2, 16] encoder每个时间步的隐藏层状态
                # print(encoder_padding_mask)

                out, attn, window_start, window_end = self.attention(
                    hidden, encoder_outs, encoder_padding_mask
                )
                for b in range(bsz):
                    start = window_start[b, 0].item()
                    end = window_end[b, 0].item()
                    attn_scores[start:end, j, b] = attn[b]

                # print(out)
                # print(attn_scores[:, j, :])
            else:
                out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def get_cached_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        input_feed = cached_state[
            "input_feed"
        ]  # can be None for decoder-only language models
        return prev_hiddens, prev_cells, input_feed

    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


# def local_score(attention_score, pt, ws, encoder_padding_mask):
#     """
#     attention_score： [seq_len, bsz] 33, 2
#     pt： [2, 1]
#     seq_len: encoder batch 中每个句子长度 [2]
#     ws: 窗口大小， 实际窗口大小： w = 2 * ws + 1, 位置：[pt - ws, pt + ws]
#     encoder_padding_mask: [32, 2]
#     源句子长度： tensor 每列的 False 个数
#     """
#     # print(pt)
#     # print("===========================================================")
#     len_padding_mask = torch.where(encoder_padding_mask == False, 1, 0)
#     # print(encoder_padding_mask)
#     seq_len = torch.sum(len_padding_mask, dim=0)  # [2] [33, 15]
#     seq_len = seq_len.unsqueeze(1)
#     # print(seq_len)
#     # tensor([[33],
#     #         [15]])
#
#     # print(seq_len)
#
#     pt = torch.where(pt + ws < seq_len, pt, seq_len - ws)
#     # print(pt)
#
#     s_len = encoder_padding_mask.size()[0]  # encoder batch 中最大句子长度 33
#
#     attention_score = attention_score.transpose(0, 1)
#     attention_score = attention_score.unsqueeze(2)
#     # print(attention_score.shape)  # [2, 33, 1]
#     # print(pt)  # [2, 1]
#     # tensor([[20],  -> 18-22
#     #         [18]], -> 16-20 )
#
#     pt = pt.unsqueeze(2)  # [2, 1, 1]
#     # print(pt.shape)
#     s = torch.arange(0, s_len).cuda()  # [33]
#     # s = torch.range(0, seq_len - 1)  # [33]
#
#     s = s.view([1, s_len, 1]).repeat([attention_score.size()[0], 1, 1])  # [2, 100, 1]
#     # print(s)
#     # print(attention_score.size()[0])  # batch size == 2
#
#     mask = torch.where((s > pt - ws) & (s < pt + ws), False, True)  # [2, 33, 1]
#     mask = torch.squeeze(mask, dim=2)
#     mask = mask.transpose(0, 1)
#     # print(mask)
#
#     return mask

def local_score(pt, ws, encoder_padding_mask, t):
    """
    attention_score： [seq_len, bsz] 33, 2
    pt： [2, 1]
    seq_len: encoder batch 中每个句子长度 [2]
    ws: 窗口大小， 实际窗口大小： w = 2 * ws + 1, 位置：[pt - ws, pt + ws]
    encoder_padding_mask: [32, 2]
    源句子长度： tensor 每列的 False 个数
    """
    # print(pt)
    # print("===========================================================")
    len_padding_mask = torch.where(encoder_padding_mask == False, 1, 0)
    # print(encoder_padding_mask)
    seq_len = torch.sum(len_padding_mask, dim=0)  # [2] [33, 15]
    seq_len = seq_len.unsqueeze(1)
    # print(seq_len)
    # tensor([[33],
    #         [15]])

    # print(seq_len)

    # T = seq_len.new_zeros()
    T = torch.full(seq_len.shape, t).cuda()
    over_len = T > seq_len
    over_len = over_len.repeat(1, encoder_padding_mask.size()[0])
    over_len = over_len.transpose(0, 1)
    # over_len = torch.where(over_len == False, 1, 0)
    # print(over_len)

    pt = torch.where(pt + ws < seq_len, pt, seq_len - ws)
    # print(pt)

    s_len = encoder_padding_mask.size()[0]  # encoder batch 中最大句子长度 33

    pt = pt.unsqueeze(2)  # [2, 1, 1]
    s = torch.arange(0, s_len).cuda()  # [33]
    s = s.view([1, s_len, 1]).repeat([encoder_padding_mask.size()[1], 1, 1])  # [2, 100, 1]

    mask = torch.where((s > pt - ws) & (s < pt + ws), False, True)  # [2, 33, 1]
    mask = torch.squeeze(mask, dim=2)
    mask = mask.transpose(0, 1)
    # print(mask)

    return mask, over_len


def local_p(h_d, fc_layer_1, fc_layer_2, seq_len):
    pt = seq_len * torch.sigmoid(fc_layer_2(torch.tanh(fc_layer_1(h_d))))
    return pt


def local_m(h_d, t):
    pt = torch.ones([h_d.size()[0], 1]) * t
    pt = pt.cuda()
    return pt


@register_model_architecture("lstm_sa", "lstm_sa")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", False)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "1")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )
    args.windows_size = getattr(args, "windows_size", 7)
