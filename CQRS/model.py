from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, List, Set


class OutOfStock(Exception):
    pass


def allocate(line: OrderLine, batches: List[Batch]) -> str:
    try:
        batch = next(b for b in sorted(batches) if b.can_allocate(line))
        batch.allocate(line)
        return batch.reference
    except StopIteration:
        raise OutOfStock(f"Out of stock for sku {line.sku}")


@dataclass(frozen=True)
class OrderLine:
    orderid: str
    sku: str
    qty: int


class Batch:
    def __init__(self, ref: str, sku: str, qty: int, eta: Optional[date]):
        self.reference = ref
        self.sku = sku
        self.eta = eta
        self._purchased_quantity = qty
        self._allocations = set()  

    def __repr__(self):
        return f"<Batch {self.reference}>"

    def __eq__(self, other):
        if not isinstance(other, Batch):
            return False
        return other.reference == self.reference

    def __hash__(self):
        return hash(self.reference)

    def __gt__(self, other):
        if self.eta is None:
            return False
        if other.eta is None:
            return True
        return self.eta > other.eta

    def allocate(self, line: OrderLine):
        if self.can_allocate(line):
            self._allocations.add(line)

    def deallocate(self, line: OrderLine):
        if line in self._allocations:
            self._allocations.remove(line)

    @property
    def allocated_quantity(self) -> int:
        return sum(line.qty for line in self._allocations)

    @property
    def available_quantity(self) -> int:
        return self._purchased_quantity - self.allocated_quantity

    def can_allocate(self, line: OrderLine) -> bool:
        return self.sku == line.sku and self.available_quantity >= line.qty
    

def main():
    
    batch1 = Batch("batch1", "RED-LAMP", 100, date.today())
    batch2 = Batch("batch2", "GREEN-LAMP", 50, date.today() + timedelta(days=1))
    batch3 = Batch("batch3", "BLUE-LAMP", 100, None)  

    print(f"Batch representation: {batch1}")
    
    print(f"Are batch1 and batch2 equal? {'Yes' if batch1 == batch2 else 'No'}")
    print(f"Hash of batch1: {hash(batch1)}")

    order_line1 = OrderLine("order1", "RED-LAMP", 10)
    order_line2 = OrderLine("order2", "GREEN-LAMP", 20)
    order_line3 = OrderLine("order3", "BLUE-LAMP", 50)
    order_line4 = OrderLine("order4", "BLUE-LAMP", 60)  

    
    for order_line in [order_line1, order_line2, order_line3, order_line4]:
        print(f"\nAllocating {order_line}...")
        try:
            allocated_batch_ref = allocate(order_line, [batch1, batch2, batch3])
            print(f"Order line {order_line} allocated to {allocated_batch_ref}")
            
            allocated_batch = next(b for b in [batch1, batch2, batch3] if b.reference == allocated_batch_ref)
            print(f"Allocated quantity of {allocated_batch_ref}: {allocated_batch.allocated_quantity}")
            print(f"Available quantity of {allocated_batch_ref}: {allocated_batch.available_quantity}")
        except OutOfStock as e:
            print(e)

    
    print(f"\nDeallocating {order_line3} from {batch3.reference}...")
    batch3.deallocate(order_line3)
    print(f"Trying to reallocate {order_line4}...")
    try:
        allocated_batch_ref = allocate(order_line4, [batch1, batch2, batch3])
        print(f"Order line {order_line4} allocated to {allocated_batch_ref}")
    except OutOfStock as e:
        print(e)


if __name__ == "__main__":
    main()