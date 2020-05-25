from typing import Type, Tuple, List, Optional, Iterable

import torch
from torch import nn

DimensionNames = List[Optional[str]]

# helper functions


def split_names(names_operation: DimensionNames) -> Tuple[DimensionNames, DimensionNames]:
    """Splits dimension names in ones before and after '*'.

    Arguments:
        names_operation {DimensionNames} -- e.g. ['N', '*', 'C'] or ['L', 'N', 'C']

    Raises:
        ValueError: If there is more than one '*' in the list.

    Returns:
        Tuple[DimensionNames, DimensionNames] -- (names_before_ast, names_after_ast), if there is no ast, then (original_list, []).

    Examples:
        ['N', '*', 'C'] -> ['N'], ['C'].
        ['L', 'N', 'C'] -> ['L', 'N', 'C'], [].
    """
    asterisks = sum(name == '*' for name in names_operation)
    if asterisks == 0:
        return names_operation[:], []
    elif asterisks == 1:
        split = names_operation.index('*')
        return names_operation[:split], names_operation[split + 1:]
    else:
        raise ValueError("More than one '*' in module dimension names.")


def names_compatible(names1: DimensionNames, names2: DimensionNames) -> bool:
    """Checks if two dimensions names are compatible. 
    To be compatible they need to have the same length, and same names for all dimensions.
    If there is None, it is compatible with any name.

    Arguments:
        names1 {DimensionNames} -- e.g. ['N', 'C', 'W', 'H']
        names2 {DimensionNames} -- e.g. [None, 'C', 'W', None]

    Returns:
        bool -- True if they are.

    Examples:
        ['N', 'C', 'W', 'H'], [None, 'C', 'W', None] -> True.
        ['N', 'C', 'W', 'H'], ['N', 'C', 'H', None] -> False.
    """
    if len(names1) != len(names2):
        return False
    else:
        for name1, name2 in zip(names1, names2):
            if name1 is not None and name2 is not None and name1 != name2:
                return False
    return True


def split_and_compare(names_in: DimensionNames, x_names: DimensionNames, layer_name: str = "") -> DimensionNames:
    """Splits x_names (name)

    Arguments:
        names_in {DimensionNames} -- Declared input dimensions names, can contain zero or one '*'.
        x_names {DimensionNames} -- Actual input dimension names, they neet do be explicit (cannot contain '*').

    Keyword Arguments:
        layer_name {str} -- Layer name for error message (default: {""}).

    Raises:
        ValueError: If there is 

    Returns:
        DimensionNames -- Dimension names from x_names as matched by '*' in names_in.

    Examples:
        ['N', '*', 'C'], ['N', 'D', 'L', C'] -> ['D', 'L'].
        ['N', '*', 'C'], ['N', 'C'] -> [].
        ['N', '*', 'C'], ['N', 'C', 'L'] -> ValueError.
    """
    error_msg = f"Layer {layer_name} requires dimensions {names_in} but got {x_names} instead."
    names_in_first, names_in_last = split_names(names_in)
    if len(names_in_first) + len(names_in_last) == len(names_in):
        # case of no '*'
        if not names_compatible(names_in, x_names):
            raise ValueError(error_msg)
        return []
    else:
        # case of '*'
        names_first = x_names[:len(names_in_first)]
        names_middle = x_names[len(names_in_first):len(x_names) - len(names_in_last)]
        names_last = x_names[len(x_names) - len(names_in_last):]
        if not names_compatible(names_in_first, names_first) or not names_compatible(names_in_last, names_last):
            raise ValueError(error_msg)
        return names_middle


def name_list_str(names: DimensionNames) -> str:
    """Turns dimension names in a string, for output purposes (__doc__, __str__, etc).
    Works the best for one-letter dimensions (otherwise it gets ambiguous).
    Turns None into _.

    Arguments:
        names {DimensionNames} -- e.g. ['N', None, 'H', 'W'].

    Returns:
        str -- e.g. 'N_HW'.
    """
    return "".join(name if name is not None else '_' for name in names)


# turning an nn.Module module instance


class NamedModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module, names_in: DimensionNames, names_out: Optional[DimensionNames] = None):
        """A wrapper that turns a normal torch.nn.Module instance into one that has typed input and output for the first argument.

        Arguments:
            module {nn.Module} -- An nn.Module instance (e.g. conv = nn.Conv2d(...), or a custom one).
            names_in {DimensionNames} -- Dimension names for the first input of forward method, e.g. ['N', 'C', 'L'], ['N', '*', 'C'].

        Keyword Arguments:
            names_out {Optional[DimensionNames]} -- Output dimension names. If not provided, uses names_in (default: {None}).
            As of now, restricted to a single output.
        """
        super().__init__()
        self.module = module
        self.names_in = names_in
        self.names_out = names_out if names_out is not None else names_in

    def forward(self, x, *args, **kwargs):
        names_middle = split_and_compare(self.names_in, list(x.names), layer_name=self.module._get_name())
        x = x.rename(None)
        x = self.module.forward(x, *args, **kwargs)
        names_out_first, names_out_last = split_names(self.names_out)
        x = x.refine_names(*names_out_first, *names_middle, *names_out_last)
        return x


# turning an nn.Module class into a named class


def name_module_class(
    old_module_class: type,
    names_in: List[DimensionNames],
    names_out: Optional[DimensionNames] = None,
    reduce_option: bool = False
) -> type:
    """Converts a nn.Module class into one with the same numerics, but that checks dimension names.

    Arguments:
        old_module_class {type} -- A class inheriting from torch.nn.Module (e.g. nn.Conv2d, or any custom).
        names_in {List[DimensionNames]} -- Dimension names for each input. E.g. [['N', 'C', 'H', 'W']] for a single input.
            It can use '*' once, and it takes any number of dimensions (including zero), e.g. [['N', '*', 'C']].

    Keyword Arguments:
        names_out {Optional[DimensionNames]} -- Output dimension names (assumes a single tensor). If None, uses the same as the first input. (default: {None})
        reduce_option {bool} -- Set True if it is a loss function and has 'reduction' option that changes output dimensions. (default: {False})

    Returns:
        type -- A class, with the nn.Module interface.
    """
    names_out = names_out if names_out is not None else names_in[0]
    signature_output = name_list_str(names_out)
    signature_inputs = ','.join(name_list_str(names_in_i) for names_in_i in names_in)
    signature = f"{signature_inputs} -> {signature_output}{' or scalar' if reduce_option else ''}"

    class NamedModule(old_module_class):  # TO FIX: mypy shows error
        """With dimension names."""
        __doc__ = f"{old_module_class.__name__} with named dimensions: {signature}.\n{old_module_class.__doc__}"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.names_in = names_in
            if reduce_option and self.reduction != 'none':
                self.names_out = []
            else:
                self.names_out = names_out

        def forward(self, *args, **kwargs):
            for x, names_in_i in zip(args, self.names_in):
                names_middle = split_and_compare(names_in_i, list(x.names), layer_name=self._get_name())
            args = [x.rename(None) for x in args]
            x = super().forward(*args, **kwargs)
            if '*' in self.names_out:
                names_out_first, names_out_last = split_names(self.names_out)
                x = x.refine_names(*names_out_first, *names_middle, *names_out_last)
            else:
                x = x.refine_names(*self.names_out)
            return x

        def __repr__(self):
            return f"{super().__repr__()} {signature}"

    NamedModule.__name__ = f"Named{old_module_class.__name__}"
    NamedModule.__qualname__ = f"Named{old_module_class.__name__}"
    return NamedModule


def name_rnn_module_class(
    old_module_class: type, names: Tuple[DimensionNames, DimensionNames], names_memory: Tuple[DimensionNames,
                                                                                              DimensionNames]
) -> type:
    """Converts a RNN nn.Module class into one with the same numerics, but that checks dimension names.

    Arguments:
        old_module_class {type} -- nn.RNN, nn.GRU, nn.LSTM or related class inheriting from torch.nn.Module.
        names {Tuple[DimensionNames, DimensionNames]} -- Dimension names of the main input=output for batch_first False then True.
            Usually (['L', 'N', 'C'], ['N', 'L', 'C']).
        names_memory {Tuple[DimensionNames,DimensionNames]} -- Dimension names of the memory cells input=output for batch_first False then True.
            Usually (['S', 'N', 'C'], ['S', 'N', 'C']) -- not a typo, both are the same.

    Returns:
        type --  A class, with the nn.Module interface.
    """
    class NamedModule(old_module_class):  # TO FIX: mypy shows error
        """With dimension names."""
        __doc__ = f"{old_module_class.__name__} with named dimensions."

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.batch_first:
                self.names = names[1]
                self.names_memory = names_memory[1]
            else:
                self.names = names[0]
                self.names_memory = names_memory[0]

            # think if that is needed, and in which format
            self.names_in = (self.names, self.names_memory)
            self.names_out = (self.names, self.names_memory)

        def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
            error_msg = f"Layer {self._get_name()} requires dimensions {self.names}, {self.names_memory} but got {x.names} instead."
            if not names_compatible(self.names, list(x.names)):
                raise ValueError(error_msg)
            if isinstance(h, torch.Tensor) and not names_compatible(self.names_memory, list(h.names)):
                raise ValueError(error_msg)
            elif isinstance(h, tuple) and not all(names_compatible(self.names_memory, list(hi.names)) for hi in h):
                raise ValueError(error_msg)

            x = x.rename(None)
            if isinstance(h, torch.Tensor):
                h = h.rename(None)
            elif isinstance(h, tuple):
                h = tuple(hi.rename(None) for hi in h)
            x, h = super().forward(x, h)
            x = x.refine_names(*self.names)
            if isinstance(h, torch.Tensor):
                h = h.refine_names(*self.names_memory)
            elif isinstance(h, tuple):
                h = tuple(hi.refine_names(*self.names_memory) for hi in h)
            return x, h

        def __repr__(self):
            return f"{super().__repr__()} {name_list_str(self.names)}"

    NamedModule.__name__ = f"Named{old_module_class.__name__}"
    NamedModule.__qualname__ = f"Named{old_module_class.__name__}"
    return NamedModule
