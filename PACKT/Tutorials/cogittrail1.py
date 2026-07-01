def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers and raise a clear error for divide-by-zero."""
    try:
        return numerator / denominator
    except ZeroDivisionError as exc:
        raise ValueError("Denominator cannot be zero.") from exc


def read_number(prompt: str) -> float:
    """Read a numeric value from user input with validation."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def main() -> None:
    try:
        num1 = read_number("Enter numerator: ")
        num2 = read_number("Enter denominator: ")
        result = safe_divide(num1, num2)
    except ValueError as error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result}")
    finally:
        print("Execution completed.")


if __name__ == "__main__":
    main()
