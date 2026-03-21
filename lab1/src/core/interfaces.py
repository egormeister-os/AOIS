from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.bit_array32 import BitArray32


class IntegerCodec(ABC):
    """Codec for converting decimal integer <-> specific binary representation."""

    @abstractmethod
    def encode(self, value: int) -> BitArray32:
        raise NotImplementedError

    @abstractmethod
    def decode(self, bits: BitArray32) -> int:
        raise NotImplementedError


class ArithmeticOperation(ABC):
    """Common operation contract used by service layer."""

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError
