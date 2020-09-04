from typing import Tuple

from torch import nn
from pytorch_named_dims.utils import DimensionNames
from pytorch_named_dims.utils import name_module_class as nmc
from pytorch_named_dims.utils import name_rnn_module_class as nmc_rnn

# Convolutional layers
Conv1d = nmc(nn.Conv1d, [['N', 'C', 'L']])
Conv2d = nmc(nn.Conv2d, [['N', 'C', 'H', 'W']])
Conv3d = nmc(nn.Conv3d, [['N', 'C', 'D', 'H', 'W']])
ConvTranspose1d = nmc(nn.ConvTranspose1d, [['N', 'C', 'L']])
ConvTranspose2d = nmc(nn.ConvTranspose2d, [['N', 'C', 'H', 'W']])
ConvTranspose3d = nmc(nn.ConvTranspose3d, [['N', 'C', 'D', 'H', 'W']])
# Unfold   NC* -> NCL
# Fold   NC* -> NC...

# Pooling layers
MaxPool1d = nmc(nn.MaxPool1d, [['N', 'C', 'L']])
MaxPool2d = nmc(nn.MaxPool2d, [['N', 'C', 'H', 'W']])
MaxPool3d = nmc(nn.MaxPool3d, [['N', 'C', 'D', 'H', 'W']])
MaxUnpool1d = nmc(nn.MaxUnpool1d, [['N', 'C', 'L']])  # in PyTorch it is NCH, which looks inconsitent
MaxUnpool2d = nmc(nn.MaxUnpool2d, [['N', 'C', 'H', 'W']])
MaxUnpool3d = nmc(nn.MaxUnpool3d, [['N', 'C', 'D', 'H', 'W']])
AvgPool1d = nmc(nn.AvgPool1d, [['N', 'C', 'L']])
AvgPool2d = nmc(nn.AvgPool2d, [['N', 'C', 'H', 'W']])
AvgPool3d = nmc(nn.AvgPool3d, [['N', 'C', 'D', 'H', 'W']])
# FractionalMaxPool2d ?
LPPool1d = nmc(nn.LPPool1d, [['N', 'C', 'L']])
LPPool2d = nmc(nn.LPPool2d, [['N', 'C', 'H', 'W']])
AdaptiveMaxPool1d = nmc(nn.AdaptiveMaxPool1d, [['N', 'C', 'L']])  # same with H
AdaptiveMaxPool2d = nmc(nn.AdaptiveMaxPool2d, [['N', 'C', 'H', 'W']])
AdaptiveMaxPool3d = nmc(nn.AdaptiveMaxPool3d, [['N', 'C', 'D', 'H', 'W']])
AdaptiveAvgPool1d = nmc(nn.AdaptiveAvgPool1d, [['N', 'C', 'L']])
AdaptiveAvgPool2d = nmc(nn.AdaptiveAvgPool2d, [['N', 'C', 'H', 'W']])
AdaptiveAvgPool3d = nmc(nn.AdaptiveAvgPool3d, [['N', 'C', 'D', 'H', 'W']])

# Padding layers
ReflectionPad1d = nmc(nn.ReflectionPad1d, [['N', 'C', 'L']])  # in docs NCW
ReflectionPad2d = nmc(nn.ReflectionPad2d, [['N', 'C', 'H', 'W']])
ReplicationPad1d = nmc(nn.ReplicationPad1d, [['N', 'C', 'L']])  # in docs NCW
ReplicationPad2d = nmc(nn.ReplicationPad2d, [['N', 'C', 'H', 'W']])
ReplicationPad3d = nmc(nn.ReplicationPad3d, [['N', 'C', 'D', 'H', 'W']])
ZeroPad2d = nmc(nn.ZeroPad2d, [['N', 'C', 'H', 'W']])
ConstantPad1d = nmc(nn.ConstantPad1d, [['N', 'C', 'L']])  # in docs NCW
ConstantPad2d = nmc(nn.ConstantPad2d, [['N', 'C', 'H', 'W']])
ConstantPad3d = nmc(nn.ConstantPad3d, [['N', 'C', 'D', 'H', 'W']])

# Nonlinar activations
# NOTE: many function have, incorrectly, N* singature
# they appear to affect names anyway, so I make them stay
ReLU = nmc(nn.ReLU, [['*']])
LogSigmoid = nmc(nn.LogSigmoid, [['*']])

# Normalization layers
BatchNorm0d = nmc(nn.BatchNorm1d, [['N', 'C']])  # not a typo, BatchNorm1d accepts NC or NCL
BatchNorm1d = nmc(nn.BatchNorm1d, [['N', 'C', 'L']])
BatchNorm2d = nmc(nn.BatchNorm2d, [['N', 'C', 'H', 'W']])
BatchNorm3d = nmc(nn.BatchNorm3d, [['N', 'C', 'D', 'H', 'W']])
GroupNorm = nmc(nn.GroupNorm, [['N', 'C', '*']])
SyncBatchNorm = nmc(nn.SyncBatchNorm, [['N', 'C', '*']])  # NC+ whatever that means
InstanceNorm1d = nmc(nn.InstanceNorm1d, [['N', 'C', 'L']])
InstanceNorm2d = nmc(nn.InstanceNorm2d, [['N', 'C', 'H', 'W']])
InstanceNorm3d = nmc(nn.InstanceNorm3d, [['N', 'C', 'D', 'H', 'W']])
LayerNorm = nmc(nn.LayerNorm, [['N', '*']])
LocalResponseNorm = nmc(nn.LocalResponseNorm, [['N', 'C', '*']])

# Recurrent layers
_rnn_names: Tuple[DimensionNames, DimensionNames] = (['L', 'N', 'C'], ['N', 'L', 'C'])  # batch_first (False, True)
_rnn_names_memory: Tuple[DimensionNames,
                         DimensionNames] = (['S', 'N', 'C'], ['S', 'N', 'C'])  # batch_first (False, True) are the same
RNN = nmc_rnn(nn.RNN, _rnn_names, _rnn_names_memory)
LSTM = nmc_rnn(nn.RNN, _rnn_names, _rnn_names_memory)
GRU = nmc_rnn(nn.RNN, _rnn_names, _rnn_names_memory)
RNNCell = nmc(nn.RNNCell, [['N', 'C'], ['N', 'C']], ['N', 'C'])
# LSTMCell NC, (NC, NC) -> NC, NC (NOTE: no 'batch_first' so cannot use nmc_rnn as it is)
RNNCell = nmc(nn.RNNCell, [['N', 'C'], ['N', 'C']], ['N', 'C'])

# Transformer layers
# TODO later (typically many inputs)

# Linear layers
# Identity - let's keep it as it is
Linear = nmc(nn.Linear, [['N', '*', 'C']])  # is N*H but used N*C for consistency
Bilinear = nmc(nn.Bilinear, [['N', '*', 'C'], ['N', '*', 'C']], ['N', '*', 'C'])  # in docs N*H N*H -> N*H

# Dropout layers
Dropout = nmc(nn.Dropout, [['*']])
Dropout2d = nmc(nn.Dropout2d, [['N', 'C', 'H', 'W']])
Dropout3d = nmc(nn.Dropout3d, [['N', 'C', 'D', 'H', 'W']])
AlphaDropout = nmc(nn.AlphaDropout, [['*']])

# Sparse layers
Embedding = nmc(nn.Embedding, [['*']], ['*', 'C'])  # is * -> *H but used C for consistency
EmbeddingBag = nmc(nn.EmbeddingBag, [['*', 'L']], ['*', 'C'])  # guessing a bit

# Distance function
# CosineSimilarity is *D* *D* -> ** (not sure if makes sense to implement it here, as D is anything)
PairwiseDistance = nmc(
    nn.PairwiseDistance, [['N', 'D'], ['N', 'D']], ['N']
)  # D, ND -> N or N1 if keep_dim (now ignored)

# Loss function
L1Loss = nmc(nn.L1Loss, [['N', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
MSELoss = nmc(nn.MSELoss, [['N', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
CrossEntropyLoss = nmc(nn.CrossEntropyLoss, [['N', 'C', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
CTCLoss = nmc(
    nn.CTCLoss, [['L', 'N', 'C'], ['N', 'S'], ['N'], ['N']], ['N'], reduce_option=True
)  # in docs T instead of L
NLLLoss = nmc(nn.NLLLoss, [['N', 'C' '*'], ['N', '*']], ['N', '*'], reduce_option=True)
PoissonNLLLoss = nmc(nn.PoissonNLLLoss, [['N', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
KLDivLoss = nmc(nn.KLDivLoss, [['N', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
BCELoss = nmc(nn.BCELoss, [['N', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
BCEWithLogitsLoss = nmc(nn.BCEWithLogitsLoss, [['N', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
MarginRankingLoss = nmc(nn.MarginRankingLoss, [['N', 'D'], ['N']], ['N'], reduce_option=True)
HingeEmbeddingLoss = nmc(nn.HingeEmbeddingLoss, [['*'], ['*']], ['*'], reduce_option=True)
MultiLabelMarginLoss = nmc(
    nn.MultiLabelMarginLoss, [['*', 'C'], ['*', 'C']], ['*'], reduce_option=True
)  # in docs '*' is N or nothing
SmoothL1Loss = nmc(nn.SmoothL1Loss, [['N', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
SoftMarginLoss = nmc(nn.SoftMarginLoss, [['*'], ['*']], ['*'], reduce_option=True)
MultiLabelSoftMarginLoss = nmc(nn.MultiLabelSoftMarginLoss, [['N', 'C'], ['N', 'C']], ['N'], reduce_option=True)
# CosineEmbeddingLoss?
# MultiMarginLoss?
TripletMarginLoss = nmc(nn.TripletMarginLoss, [['N', 'D'], ['N', 'D'], ['N', 'D']], ['N'], reduce_option=True)

# Vision layers
PixelShuffle = nmc(nn.PixelShuffle, [['N', 'C', 'H', 'W']])  # in docs NLHW -> NCHW
Upsample = nmc(nn.Upsample, [['N', 'C', '*']])
Upsample1d = nmc(nn.Upsample, [['N', 'C', 'L']])  # experimenting
Upsample2d = nmc(nn.Upsample, [['N', 'C', 'H', 'W']])
Upsample3d = nmc(nn.Upsample, [['N', 'C', 'D', 'H', 'W']])
UpsamplingNearest2d = nmc(nn.UpsamplingNearest2d, [['N', 'C', 'H', 'W']])
UpsamplingBilinear2d = nmc(nn.UpsamplingBilinear2d, [['N', 'C', 'H', 'W']])
