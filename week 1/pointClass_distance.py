import math

class Point:
    """
    Represents a point in a two-dimensional space with x and y coordinates.
    """
    def __init__(self, x, y):
        """Initializes the point with x and y attributes."""
        self.x = x  # Attribute 1: x-coordinate
        self.y = y  # Attribute 2: y-coordinate
        
    def distance(self, other_point):
        """
        Calculates the Euclidean distance between this point and another point.
        
        :param other_point: Another Point object.
        :return: The distance as a float.
        """
        # (x2 - x1)
        dx = other_point.x - self.x
        
        # (y2 - y1)
        dy = other_point.y - self.y
        
        # distance = sqrt((dx^2) + (dy^2))
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance

    def __str__(self):
        """Provides a user-friendly string representation of the object."""
        return f"Point({self.x}, {self.y})"

# --- Usage Example ---

# 1. Create two Point objects (instances)
p1 = Point(x=1, y=2)
p2 = Point(x=4, y=6)

print(f"Point 1: {p1}")
print(f"Point 2: {p2}")

# 2. Calculate the distance between them
# Expected calculation: sqrt((4-1)^2 + (6-2)^2) = sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.0
distance_result = p1.distance(p2)

print(f"\nThe distance between P1 and P2 is: {distance_result:.2f}")