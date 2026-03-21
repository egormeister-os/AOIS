# Lab 1: Bit-Level Calculator

## Student
- Full name: `Mukhamedzianav Egor`
- Group: `421701`
- Variant: `D (Excess-3 BCD)`

## Project Goal
This project implements a bit-level calculator for the first laboratory work.
All core data is represented as fixed 32-bit arrays of `0` and `1`, and the
main conversions and arithmetic operations are implemented manually without
using helper libraries for binary or IEEE-754 encoding.

## Implemented Functionality

### Integer representations
- Convert decimal integers to:
  - sign-magnitude code
  - ones' complement
  - two's complement
- Convert these representations back to decimal for verification

### Integer arithmetic
- Addition in two's complement
- Subtraction in two's complement using `A + (-B)`
- Multiplication in sign-magnitude
- Division in sign-magnitude with binary fractional part and decimal precision

### IEEE-754 float32
- Manual encoding of decimal values to IEEE-754 single precision (32-bit)
- Manual decoding of 32-bit float representation back to decimal
- Operations:
  - addition
  - subtraction
  - multiplication
  - division
- Support for special values and cases:
  - `+0`, `-0`
  - `+inf`, `-inf`
  - `NaN`
  - subnormal numbers

### Excess-3 BCD
- Encoding of decimal numbers to Excess-3 tetrads
- Addition of two decimal numbers in Excess-3 code

### CLI
- Interactive command-line menu for all implemented laboratory tasks

## Architecture
- `src/core` - base 32-bit container and common interfaces
- `src/converters` - integer representation codecs
- `src/operations` - arithmetic engines
- `src/services` - facade for calling all operations from one place
- `src/ui` - CLI and output formatting
- `tests` - unit and branch tests
- `report` - theory and examples for the lab report

## Project Tree
```text
lab1/
├─ run.py
├─ requirements.txt
├─ src/
│  ├─ core/
│  ├─ converters/
│  ├─ operations/
│  ├─ services/
│  └─ ui/
├─ tests/
│  ├─ core/
│  ├─ converters/
│  ├─ operations/
│  ├─ services/
│  └─ ui/
└─ report/
   ├─ theory.md
   └─ examples.md
```

## Requirements
- Python 3.14+
- Packages from `requirements.txt`

Install dependencies into the local virtual environment:

```bash
venv/bin/python -m pip install -r requirements.txt
```

## Run
Launch the CLI from the project root:

```bash
venv/bin/python run.py
```

## Tests
Run the full test suite:

```bash
venv/bin/python -m pytest -q
```

Current result:
- `59 passed`

## Coverage
Measure coverage for source files only:

```bash
venv/bin/python -m coverage run --source=src -m pytest -q
venv/bin/python -m coverage report -m
```

Current source coverage:
- `98%`

## Notes
- All main operations work with `BitArray32` or values converted to it.
- The project is no longer a scaffold: the required laboratory functionality is
  implemented and covered with tests.
- The report files can be filled with examples from the CLI or from direct
  service-layer calls.
