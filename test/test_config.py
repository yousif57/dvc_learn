import pytest

class NotInRange(Exception):
    def __init__(self, massege="value not in range"):
        #self.input_ = input_
        self.massege = massege
        super().__init__(self.massege)

def test_generic():
    a = 2
    with pytest.raises(NotInRange):
        if a not in range(10,20):
            raise NotInRange
    