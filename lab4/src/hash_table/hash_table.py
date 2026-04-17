"""Hash table implementation with linked-list collision resolution."""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from hash_table.exceptions import InvalidKeyError
from hash_table.linked_list import LinkedListChain

ValueT = TypeVar("ValueT")


class HashTable(Generic[ValueT]):
    """Hash table with CRUD operations and chaining via linked lists.

    The hash function follows the task formula: ``h(V) = V mod H + B``.
    ``H`` is the table capacity and ``B`` is a configurable base offset.
    """

    def __init__(self, size: int, base: int = 0) -> None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("Table size H must be an integer")
        if size <= 0:
            raise ValueError("Table size H must be positive")
        if isinstance(base, bool) or not isinstance(base, int):
            raise ValueError("Base offset B must be an integer")

        self._size = size
        self._base = base
        self._buckets = [LinkedListChain[ValueT]() for _ in range(size)]
        self._items_count = 0

    @property
    def size(self) -> int:
        """Return H, the number of buckets in the table."""

        return self._size

    @property
    def base(self) -> int:
        """Return B, the base offset used by the hash function."""

        return self._base

    @property
    def load_factor(self) -> float:
        return self._items_count / self._size

    def __len__(self) -> int:
        return self._items_count

    def __contains__(self, key: int) -> bool:
        self._validate_key(key)
        return self._bucket_for(key).contains(key)

    def __iter__(self) -> Iterator[tuple[int, ValueT]]:
        return self.items()

    def hash(self, key: int) -> int:
        """Return ``h(V) = V mod H + B`` for an integer key."""

        self._validate_key(key)
        return key % self._size + self._base

    def create(self, key: int, value: ValueT) -> None:
        """Create a new key-value pair.

        Raises:
            DuplicateKeyError: if the key already exists.
        """

        self._validate_key(key)
        self._bucket_for(key).insert(key, value)
        self._items_count += 1

    def read(self, key: int) -> ValueT:
        """Return the value for an existing key."""

        self._validate_key(key)
        return self._bucket_for(key).get(key)

    def update(self, key: int, value: ValueT) -> None:
        """Update an existing key-value pair."""

        self._validate_key(key)
        self._bucket_for(key).replace(key, value)

    def delete(self, key: int) -> ValueT:
        """Delete an existing key and return the removed value."""

        self._validate_key(key)
        removed_value = self._bucket_for(key).remove(key)
        self._items_count -= 1
        return removed_value

    def items(self) -> Iterator[tuple[int, ValueT]]:
        """Iterate over all key-value pairs in the table."""

        for bucket in self._buckets:
            yield from bucket

    def bucket_lengths(self) -> tuple[int, ...]:
        """Return bucket chain lengths, useful for diagnostics and tests."""

        return tuple(len(bucket) for bucket in self._buckets)

    def buckets(self) -> tuple[tuple[tuple[int, ValueT], ...], ...]:
        """Return immutable snapshots of all collision chains."""

        return tuple(tuple(bucket) for bucket in self._buckets)

    def _bucket_for(self, key: int) -> LinkedListChain[ValueT]:
        return self._buckets[self.hash(key) - self._base]

    @staticmethod
    def _validate_key(key: int) -> None:
        if isinstance(key, bool) or not isinstance(key, int):
            raise InvalidKeyError("Hash function h(V) expects an integer key V")
