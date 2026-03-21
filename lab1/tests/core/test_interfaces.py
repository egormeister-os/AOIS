import pytest

from src.core.bit_array32 import BitArray32
from src.core.interfaces import ArithmeticOperation, IntegerCodec


class DummyCodec(IntegerCodec):
    def encode(self, value: int) -> BitArray32:
        return BitArray32()

    def decode(self, bits: BitArray32) -> int:
        return 0


class DummyOperation(ArithmeticOperation):
    def execute(self, *args, **kwargs):
        return None


def test_interface_base_methods_raise_not_implemented() -> None:
    codec = DummyCodec()
    operation = DummyOperation()

    with pytest.raises(NotImplementedError):
        IntegerCodec.encode(codec, 1)

    with pytest.raises(NotImplementedError):
        IntegerCodec.decode(codec, BitArray32())

    with pytest.raises(NotImplementedError):
        ArithmeticOperation.execute(operation)
