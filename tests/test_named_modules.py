import pytest
import torch
from torch import nn
from pytorch_named_dims import NamedModuleWrapper, name_module_class, name_rnn_module_class

# good
x_input_1 = torch.rand((4, 3, 2, 2), names=('N', 'C', 'H', 'W'))
x_input_2 = torch.rand((4, 3, 2, 2), names=(None, 'C', None, 'W'))
# bad
x_input_3 = torch.rand((4, 3, 2, 2), names=('N', 'C', 'W', 'H'))
x_input_4 = torch.rand((4, 3, 2), names=('N', 'C', 'H'))


def test_named_instance():
    named_conv_2d = NamedModuleWrapper(nn.Conv2d(3, 5, kernel_size=3, padding=1), ['N', 'C', 'H', 'W'])
    output_1 = named_conv_2d(x_input_1)
    assert output_1.names == ('N', 'C', 'H', 'W')
    output_2 = named_conv_2d(x_input_2)
    assert output_2.names == ('N', 'C', 'H', 'W')
    with pytest.raises(ValueError):
        assert named_conv_2d(x_input_3)
    with pytest.raises(ValueError):
        assert named_conv_2d(x_input_4)


def test_name_module_class():
    NamedConv2d = name_module_class(nn.Conv2d, [['N', 'C', 'H', 'W']])
    named_conv_2d = NamedConv2d(3, 5, kernel_size=3, padding=1)
    output_1 = named_conv_2d(x_input_1)
    assert output_1.names == ('N', 'C', 'H', 'W')
    output_2 = named_conv_2d(x_input_2)
    assert output_2.names == ('N', 'C', 'H', 'W')
    with pytest.raises(ValueError):
        assert named_conv_2d(x_input_3)
    with pytest.raises(ValueError):
        assert named_conv_2d(x_input_4)


def test_name_module_class_name_change():
    NamedConv2d = name_module_class(nn.Conv2d, [['N', 'C', 'H', 'W']], ['B', 'C', 'X', 'Y'])
    named_conv_2d = NamedConv2d(3, 5, kernel_size=3, padding=1)
    output_2 = named_conv_2d(x_input_2)
    assert output_2.names == ('B', 'C', 'X', 'Y')


def test_name_module_class_encapsulation():
    NamedConv2d = name_module_class(nn.Conv2d, [['N', 'C', 'H', 'W']])
    named_conv_2d = NamedConv2d(3, 5, kernel_size=3, padding=1)
    assert not hasattr(named_conv_2d, 'in_names')


def test_asterisk():
    x_input = torch.randint(2, (4, 3)).refine_names('N', 'X')
    NamedEmbedding = name_module_class(nn.Embedding, [['*']], ['*', 'C'])
    named_emb = NamedEmbedding(2, 5)
    output = named_emb(x_input)
    assert output.names == ('N', 'X', 'C')


def test_name_module_class_str():
    NamedConv2d = name_module_class(nn.Conv2d, [['N', 'C', 'H', 'W']])
    named_conv_2d = NamedConv2d(3, 5, kernel_size=3, padding=1)
    assert str(named_conv_2d) == "NamedConv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) NCHW -> NCHW"


def test_name_rnn_module_class_lstm():
    NamedLSTM = name_rnn_module_class(nn.LSTM, (['L', 'N', 'C'], ['N', 'L', 'C']), (['S', 'N', 'C'], ['S', 'N', 'C']))
    named_lstm = NamedLSTM(3, 5, batch_first=False)
    x_0 = torch.rand((4, 2, 3), names=('L', 'N', 'C'))
    h_0 = torch.rand((1, 2, 5), names=('S', 'N', 'C'))
    c_0 = torch.rand((1, 2, 5), names=('S', 'N', 'C'))
    x_1, (h_1, c_1) = named_lstm(x_0, (h_0, c_0))
    assert x_1.names == ('L', 'N', 'C')
    assert h_1.names == ('S', 'N', 'C')
    assert c_1.names == ('S', 'N', 'C')
    x_1, (h_1, c_1) = named_lstm(x_0)
    assert x_1.names == ('L', 'N', 'C')
    assert h_1.names == ('S', 'N', 'C')
    assert c_1.names == ('S', 'N', 'C')

    named_lstm_bf = NamedLSTM(3, 5, batch_first=True)
    x_0_bf = torch.rand((2, 4, 3), names=('N', 'L', 'C'))
    # order of (h, c) remains the same
    x_1, (h_1, c_1) = named_lstm_bf(x_0_bf, (h_0, c_0))
    assert x_1.names == ('N', 'L', 'C')
    assert h_1.names == ('S', 'N', 'C')
    assert c_1.names == ('S', 'N', 'C')
    x_1, (h_1, c_1) = named_lstm_bf(x_0_bf)
    assert x_1.names == ('N', 'L', 'C')
    assert h_1.names == ('S', 'N', 'C')
    assert c_1.names == ('S', 'N', 'C')

    with pytest.raises(ValueError):
        assert named_lstm(x_0_bf)
    with pytest.raises(ValueError):
        assert named_lstm(x_0_bf, (h_0, c_0))
    with pytest.raises(ValueError):
        assert named_lstm_bf(x_0)


def test_name_module_class_bilinear():
    x_1 = torch.rand((4, 2, 3), names=('N', 'X', 'C'))
    x_2 = torch.rand((4, 2, 3), names=('N', 'X', 'C'))
    x_3 = torch.rand((4, 2, 3), names=('N', 'C', 'X'))
    NamedBilinear = name_module_class(nn.Bilinear, [['N', '*', 'C'], ['N', '*', 'C']], ['N', '*', 'C'])
    named_bilinear = NamedBilinear(3, 3, 5)
    output = named_bilinear(x_1, x_2)
    assert output.names == ('N', 'X', 'C')
    with pytest.raises(ValueError):
        assert named_bilinear(x_1, x_3)


def test_name_module_class_loss_reduce():
    preds_1 = torch.rand((4, 4), names=('N', 'C'))
    preds_1_wrong = torch.rand((4, 4), names=('C', 'N'))
    labels_1 = torch.randint(4, (4, )).refine_names('N')
    preds_2 = torch.rand((4, 3, 5, 6), names=('N', 'C', 'X', 'Y'))
    labels_2 = torch.randint(3, (4, 5, 6)).refine_names('N', 'X', 'Y')
    NamedCELoss = name_module_class(nn.CrossEntropyLoss, [['N', 'C', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
    named_celoss = NamedCELoss(reduction='mean')
    output_1 = named_celoss(preds_1, labels_1)
    assert output_1.names == ()
    output_2 = named_celoss(preds_2, labels_2)
    assert output_2.names == ()
    with pytest.raises(ValueError):
        assert named_celoss(preds_1_wrong, labels_1)


def test_name_module_class_loss_reduce_set_to_none():
    preds_1 = torch.rand((4, 4), names=('N', 'C'))
    preds_1_wrong = torch.rand((4, 4), names=('C', 'N'))
    labels_1 = torch.randint(4, (4, )).refine_names('N')
    preds_2 = torch.rand((4, 3, 5, 6), names=('N', 'C', 'X', 'Y'))
    labels_2 = torch.randint(3, (4, 5, 6)).refine_names('N', 'X', 'Y')
    NamedCELoss = name_module_class(nn.CrossEntropyLoss, [['N', 'C', '*'], ['N', '*']], ['N', '*'], reduce_option=True)
    named_celoss = NamedCELoss(reduction='none')
    output_1 = named_celoss(preds_1, labels_1)
    assert output_1.names == ('N', )
    output_2 = named_celoss(preds_2, labels_2)
    assert output_2.names == ('N', 'X', 'Y')
    with pytest.raises(ValueError):
        assert named_celoss(preds_1_wrong, labels_1)
