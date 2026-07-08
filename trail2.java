import java.util.Scanner;

class LargestSmallest {
	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);

		System.out.print("Enter how many numbers you want to check: ");
		int count = scanner.nextInt();

		if (count <= 0) {
			System.out.println("Please enter a positive count.");
			scanner.close();
			return;
		}

		System.out.print("Enter number 1: ");
		int firstNumber = scanner.nextInt();
		int largest = firstNumber;
		int smallest = firstNumber;

		for (int i = 2; i <= count; i++) {
			System.out.print("Enter number " + i + ": ");
			int number = scanner.nextInt();

			if (number > largest) {
				largest = number;
			}

			if (number < smallest) {
				smallest = number;
			}
		}

		System.out.println("Largest number: " + largest);
		System.out.println("Smallest number: " + smallest);

		scanner.close();
	}
}
