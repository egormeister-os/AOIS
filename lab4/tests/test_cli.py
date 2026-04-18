import subprocess
import sys
from pathlib import Path

from hash_table.cli import main
from hash_table.demo_data import save_demo_table
from hash_table.storage import JsonHashTableStorage


def test_cli_can_run_full_crud_flow(tmp_path: Path, capsys) -> None:
    state = tmp_path / "state.json"

    assert main(["--state", str(state), "init", "--size", "3", "--base", "10"]) == 0
    assert "Created empty table" in capsys.readouterr().out

    assert main(["--state", str(state), "create", "1", "one"]) == 0
    assert "h(V)=11, bucket=1" in capsys.readouterr().out

    assert main(["--state", str(state), "create", "4", "four"]) == 0
    assert "bucket=1" in capsys.readouterr().out

    assert main(["--state", str(state), "read", "1"]) == 0
    assert "value='one'" in capsys.readouterr().out

    assert main(["--state", str(state), "update", "1", "updated"]) == 0
    assert "Updated key=1" in capsys.readouterr().out

    assert main(["--state", str(state), "hash", "4"]) == 0
    assert "key=4: h(V)=11, bucket=1" in capsys.readouterr().out

    assert main(["--state", str(state), "list"]) == 0
    output = capsys.readouterr().out
    assert "Table H=3, B=10, count=2" in output
    assert "bucket 0 (h=10): empty" in output
    assert "bucket 1 (h=11): 4='four' -> 1='updated'" in output
    assert "| ID | C | T | Po | Pi      |" in output
    assert "| 4  | 1 | 0 | #2 | four    |" in output
    assert "| 1  | 1 | 1 | -  | updated |" in output
    assert "| U |" not in output
    assert "| L |" not in output
    assert "| D |" not in output

    assert main(["--state", str(state), "delete", "4"]) == 0
    assert "removed value='four'" in capsys.readouterr().out

    table = JsonHashTableStorage(state).load()
    assert dict(table.items()) == {1: "updated"}


def test_cli_reports_missing_state_file(tmp_path: Path, capsys) -> None:
    state = tmp_path / "missing.json"

    assert main(["--state", str(state), "read", "1"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_cli_init_requires_force_to_overwrite(tmp_path: Path, capsys) -> None:
    state = tmp_path / "state.json"

    assert main(["--state", str(state), "init", "--size", "3"]) == 0
    capsys.readouterr()

    assert main(["--state", str(state), "init", "--size", "5"]) == 1
    assert "already exists" in capsys.readouterr().err

    assert main(["--state", str(state), "init", "--size", "5", "--force"]) == 0
    assert "H=5" in capsys.readouterr().out


def test_cli_reports_table_errors(tmp_path: Path, capsys) -> None:
    state = tmp_path / "state.json"

    assert main(["--state", str(state), "init", "--size", "3"]) == 0
    capsys.readouterr()
    assert main(["--state", str(state), "create", "1", "one"]) == 0
    capsys.readouterr()

    assert main(["--state", str(state), "create", "1", "duplicate"]) == 1
    assert "already exists" in capsys.readouterr().err

    assert main(["--state", str(state), "delete", "9"]) == 1
    assert "was not found" in capsys.readouterr().err


def test_cli_demo_command_populates_state_file(tmp_path: Path, capsys) -> None:
    state = tmp_path / "demo.json"

    assert main(["--state", str(state), "demo"]) == 0
    output = capsys.readouterr().out
    assert "Saved demo table" in output

    table = JsonHashTableStorage(state).load()
    assert table.bucket_lengths() == (0, 3, 2, 0, 0)

    assert main(["--state", str(state), "demo"]) == 1
    assert "already exists" in capsys.readouterr().err


def test_demo_data_helper_saves_intentional_collisions(tmp_path: Path) -> None:
    state = tmp_path / "demo.json"

    table = save_demo_table(state)

    assert table.size == 5
    assert table.base == 100
    assert table.bucket_lengths() == (0, 3, 2, 0, 0)
    assert JsonHashTableStorage(state).load().bucket_lengths() == (0, 3, 2, 0, 0)


def test_cli_scripts_can_be_executed_directly(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]

    root_state = tmp_path / "root_cli.json"
    root_result = subprocess.run(
        [sys.executable, "cli.py", "--state", str(root_state), "demo"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert root_result.returncode == 0, root_result.stderr
    assert "Saved demo table" in root_result.stdout

    package_state = tmp_path / "package_cli.json"
    package_result = subprocess.run(
        [sys.executable, "cli.py", "--state", str(package_state), "demo"],
        cwd=root / "src" / "hash_table",
        capture_output=True,
        check=False,
        text=True,
    )

    assert package_result.returncode == 0, package_result.stderr
    assert "Saved demo table" in package_result.stdout
