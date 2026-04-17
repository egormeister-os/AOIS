import pytest

from hash_table import (
    DuplicateKeyError,
    HashTable,
    InvalidKeyError,
    KeyNotFoundError,
)


def test_create_read_update_delete_roundtrip() -> None:
    table = HashTable[str](size=5, base=10)

    assert len(table) == 0
    assert table.load_factor == 0
    assert table.size == 5
    assert table.base == 10
    assert table.hash(7) == 12

    table.create(7, "seven")

    assert 7 in table
    assert len(table) == 1
    assert table.load_factor == pytest.approx(0.2)
    assert table.read(7) == "seven"

    table.update(7, "updated")

    assert table.read(7) == "updated"
    assert table.delete(7) == "updated"
    assert len(table) == 0
    assert 7 not in table


def test_collision_resolution_uses_linked_list_chains() -> None:
    table = HashTable[str](size=3)

    table.create(1, "one")
    table.create(4, "four")
    table.create(7, "seven")

    assert table.hash(1) == table.hash(4) == table.hash(7)
    assert table.bucket_lengths() == (0, 3, 0)
    assert table.read(1) == "one"
    assert table.read(4) == "four"
    assert table.read(7) == "seven"
    assert dict(table.items()) == {1: "one", 4: "four", 7: "seven"}
    assert dict(iter(table)) == {1: "one", 4: "four", 7: "seven"}


def test_delete_from_head_middle_and_tail_of_collision_chain() -> None:
    table = HashTable[str](size=2)

    table.create(1, "tail")
    table.create(3, "middle")
    table.create(5, "head")

    assert table.bucket_lengths() == (0, 3)
    assert table.delete(5) == "head"
    assert table.bucket_lengths() == (0, 2)
    assert table.read(1) == "tail"
    assert table.read(3) == "middle"

    assert table.delete(1) == "tail"
    assert table.bucket_lengths() == (0, 1)
    assert table.read(3) == "middle"

    assert table.delete(3) == "middle"
    assert table.bucket_lengths() == (0, 0)
    assert len(table) == 0


def test_create_rejects_duplicate_key() -> None:
    table = HashTable[int](size=4)
    table.create(9, 100)

    with pytest.raises(DuplicateKeyError, match="already exists"):
        table.create(9, 200)

    assert table.read(9) == 100
    assert len(table) == 1


@pytest.mark.parametrize("operation", ["read", "update", "delete"])
def test_missing_key_operations_raise_key_not_found(operation: str) -> None:
    table = HashTable[str](size=4)

    with pytest.raises(KeyNotFoundError, match="was not found"):
        if operation == "read":
            table.read(1)
        elif operation == "update":
            table.update(1, "value")
        else:
            table.delete(1)


@pytest.mark.parametrize("size", [0, -1, 2.5, True])
def test_table_size_must_be_positive_integer(size: object) -> None:
    with pytest.raises(ValueError, match="size H"):
        HashTable(size=size)  # type: ignore[arg-type]


@pytest.mark.parametrize("base", [1.5, False])
def test_base_offset_must_be_integer(base: object) -> None:
    with pytest.raises(ValueError, match="Base offset B"):
        HashTable(size=3, base=base)  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["1", 1.5, True])
def test_key_must_be_integer_for_hash_function(key: object) -> None:
    table = HashTable[str](size=3)

    with pytest.raises(InvalidKeyError, match="integer key"):
        table.create(key, "value")  # type: ignore[arg-type]


def test_negative_integer_keys_are_supported_by_modulo_hash() -> None:
    table = HashTable[str](size=5, base=100)

    table.create(-1, "minus one")

    assert table.hash(-1) == 104
    assert table.read(-1) == "minus one"
