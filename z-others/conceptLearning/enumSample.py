from enum import Enum


# Define an Enum class
class OrderStatus(Enum):
    PENDING = 1
    PROCESSING = 2
    SHIPPED = 3
    DELIVERED = 4

    # Access by attribute


status = OrderStatus.PENDING
print(status)  # Output: OrderStatus.PENDING

# Access by value
print(OrderStatus(2))  # Output: OrderStatus.PROCESSING

# Access by name (string)
print(OrderStatus["SHIPPED"])  # Output: OrderStatus.SHIPPED

current = OrderStatus.DELIVERED
print(current.name)  # Output: 'DELIVERED'
print(current.value)  # Output: 4

for status in OrderStatus:
    print(f"{status.name} -> {status.value}")
