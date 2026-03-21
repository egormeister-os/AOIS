from __future__ import annotations


class BitArray32:
    """Mutable 32-bit container used as the canonical internal representation."""

    SIZE = 32

    def __init__(self, bits: list[int] | None = None) -> None:
        if bits is None:
            self._bits = [0] * self.SIZE
            return
        if len(bits) != self.SIZE or any(bit not in (0, 1) for bit in bits):
            raise ValueError("BitArray32 expects exactly 32 bits (0/1).")
        self._bits = bits[:]

    @property
    def bits(self) -> list[int]:
        return self._bits

    def copy(self) -> "BitArray32":
        return BitArray32(self._bits)

    def invert(self) -> "BitArray32":
        result = self.copy()
        for i in range(self.SIZE):
            result[i] = 1 - result[i]
        return result

    def add_one(self) -> "BitArray32":
        result = self.copy()
        for i in range(self.SIZE - 1, -1, -1):
            if result[i] == 0:
                result[i] = 1
                break
            result[i] = 0
        return result

    def __getitem__(self, idx: int) -> int:
        return self._bits[idx]

    def __setitem__(self, idx: int, value: int) -> None:
        if value not in (0, 1):
            raise ValueError("Bit value must be 0 or 1.")
        self._bits[idx] = value

    def __len__(self) -> int:
        return self.SIZE

    def __iter__(self):
        return iter(self._bits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitArray32):
            return NotImplemented
        return self._bits == other._bits

    def __repr__(self) -> str:
        return f"BitArray32(bits={self.__str__()!r})"

    def is_zero(self) -> bool:
        return all(bit == 0 for bit in self._bits)

    def __str__(self) -> str:
        return "".join(str(bit) for bit in self._bits)
