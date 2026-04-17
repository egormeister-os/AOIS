"""Domain-specific exceptions for the hash table."""


class HashTableError(Exception):
    """Base class for all hash table errors."""


class InvalidKeyError(HashTableError, TypeError):
    """Raised when a key cannot be used by the configured hash function."""


class DuplicateKeyError(HashTableError, KeyError):
    """Raised when a create operation receives an existing key."""


class KeyNotFoundError(HashTableError, KeyError):
    """Raised when a requested key does not exist in the table."""
