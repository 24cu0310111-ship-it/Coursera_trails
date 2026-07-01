"""Exception Handling Demo

A demonstration of proper exception handling patterns in Python.
"""

import logging
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def read_file_safely(filename: str, encoding: str = "utf-8") -> Optional[str]:
    """Read file contents with comprehensive exception handling.

    Args:
        filename: Path to the file to read.
        encoding: Character encoding to use (default: utf-8).

    Returns:
        File contents as string, or None if reading failed.
    """
    try:
        with open(filename, "r", encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File '{filename}' not found")
    except PermissionError:
        logger.error(f"Permission denied to read '{filename}'")
    except UnicodeDecodeError:
        logger.error(f"Cannot decode '{filename}' with encoding '{encoding}'")
    except OSError as e:
        logger.error(f"OS error reading '{filename}': {e}")
    return None


def get_list_element(lst: list[Any], index: int) -> tuple[bool, Any]:
    """Safely retrieve an element from a list by index.

    Args:
        lst: The list to access.
        index: The index of the element to retrieve.

    Returns:
        A tuple of (success: bool, value: Any).
        If successful, value contains the element.
        If failed, value contains the error message.
    """
    if not isinstance(index, int):
        return (False, "Index must be an integer")

    try:
        return (True, lst[index])
    except IndexError:
        return (False, f"Index {index} out of range for list of length {len(lst)}")
    except TypeError:
        return (False, "First argument must be a sequence")


def divide_safely(numerator: float, denominator: float) -> tuple[bool, float]:
    """Perform division with zero-division handling.

    Args:
        numerator: The number to divide.
        denominator: The number to divide by.

    Returns:
        A tuple of (success: bool, result: float).
    """
    try:
        return (True, numerator / denominator)
    except ZeroDivisionError:
        return (False, 0.0)


def main() -> None:
    """Run the exception handling demonstration."""
    print("=== Exception Handling Demo ===\n")

    # File reading examples
    print("--- File Reading ---")
    content = read_file_safely("nonexistent.txt")
    if content is None:
        print("(File read failed as expected)\n")

    # List access examples
    print("--- List Access ---")
    my_list = [1, 2, 3]

    success, value = get_list_element(my_list, 1)
    print(f"Index 1: {value}" if success else f"Error: {value}")

    success, value = get_list_element(my_list, 10)
    print(f"Error: {value}" if not success else f"Value: {value}")

    success, value = get_list_element(my_list, "a")
    print(f"Error: {value}" if not success else f"Value: {value}")

    # Division examples
    print("\n--- Division ---")
    success, result = divide_safely(10, 2)
    print(f"10 / 2 = {result}" if success else "Division failed")

    success, result = divide_safely(10, 0)
    print(f"10 / 0 = Error (division by zero)" if not success else f"Result: {result}")


if __name__ == "__main__":
    main()