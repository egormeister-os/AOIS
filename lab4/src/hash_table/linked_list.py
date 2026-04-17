"""Linked-list chains used to resolve hash collisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar

from hash_table.exceptions import DuplicateKeyError, KeyNotFoundError

ValueT = TypeVar("ValueT")


@dataclass(slots=True)
class _Node(Generic[ValueT]):
    key: int
    value: ValueT
    next: _Node[ValueT] | None = None


class LinkedListChain(Generic[ValueT]):
    """A bucket chain that stores key-value pairs in a singly linked list."""

    def __init__(self) -> None:
        self._head: _Node[ValueT] | None = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[tuple[int, ValueT]]:
        current = self._head
        while current is not None:
            yield current.key, current.value
            current = current.next

    def contains(self, key: int) -> bool:
        return self._find_node(key) is not None

    def insert(self, key: int, value: ValueT) -> None:
        if self.contains(key):
            raise DuplicateKeyError(f"Key {key} already exists")

        self._head = _Node(key=key, value=value, next=self._head)
        self._size += 1

    def get(self, key: int) -> ValueT:
        node = self._find_node(key)
        if node is None:
            raise KeyNotFoundError(f"Key {key} was not found")

        return node.value

    def replace(self, key: int, value: ValueT) -> None:
        node = self._find_node(key)
        if node is None:
            raise KeyNotFoundError(f"Key {key} was not found")

        node.value = value

    def remove(self, key: int) -> ValueT:
        previous: _Node[ValueT] | None = None
        current = self._head

        while current is not None:
            if current.key == key:
                if previous is None:
                    self._head = current.next
                else:
                    previous.next = current.next

                self._size -= 1
                return current.value

            previous = current
            current = current.next

        raise KeyNotFoundError(f"Key {key} was not found")

    def _find_node(self, key: int) -> _Node[ValueT] | None:
        current = self._head
        while current is not None:
            if current.key == key:
                return current
            current = current.next

        return None
