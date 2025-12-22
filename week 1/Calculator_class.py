class Calculator:
    """
    A simple utility class to perform basic arithmetic operations.
    Note: A dedicated __init__ method is not needed as there are no instance attributes (self.x) 
    that need to be stored upon creation.
    """
    
    def add(self, a, b):
        """Returns the sum of two numbers."""
        return a + b
        
    def subtract(self, a, b):
        """Returns the difference of two numbers (a - b)."""
        return a - b
        
    def multiply(self, a, b):
        """Returns the product of two numbers."""
        return a * b
        
    def divide(self, a, b):
        """
        Returns the result of division (a / b).
        Includes basic error handling for division by zero.
        """
        if b == 0:
            return "Error: Cannot divide by zero"
        return a / b

# --- Usage Example ---

# 1. Create a Calculator object
calc = Calculator()

# 2. Perform operations
sum_result = calc.add(10, 5)
diff_result = calc.subtract(20, 7)
prod_result = calc.multiply(4, 6)
div_result = calc.divide(15, 3)
zero_div_result = calc.divide(10, 0)

print("\n--- Calculator Results ---")
print(f"10 + 5 = {sum_result}")
print(f"20 - 7 = {diff_result}")
print(f"4 * 6 = {prod_result}")
print(f"15 / 3 = {div_result}")
print(f"10 / 0 = {zero_div_result}")