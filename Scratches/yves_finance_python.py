class FinancialInstrument(object):
    author = "Yves Hilpisch"

    def __init__(self, symbol, price):
        self.symbol = symbol
        self.price = price


FinancialInstrument.author
aapl = FinancialInstrument("AAPL", 100)
aapl.symbol
aapl.price


class FinancialInstrument(FinancialInstrument):
    def get_price(self):
        return self.price

    def set_price(self, price):
        self.price = price


fi = FinancialInstrument("AAPL", 100)
fi.get_price()
fi.set_price(105)
fi.get_price()
fi.price


class __FinancialInstrument(object):
    def __init__(self, symbol, price):
        self.symbol = symbol
        self.__price = price

    def get_price(self):
        return self.__price

    def set_price(self, price):
        self.__price = price


fi = __FinancialInstrument("AAPL", 100)
fi._FinancialInstrument
fi.__FinancialInstrument__price
fi.__FinancialInstrument__price = 105
fi.__FinancialInstrument__price

fi.get_price()
fi.symbol


a = ["pretzels", "carrots", "arugula", "bacon"]
for _ in range(len(a)):
    # print(_)
    for i in range(1, len(a)):
        print(_, i, a[i], a[i - 1])


def bubble_sort(a):
    for _ in range(len(a)):
        for i in range(1, len(a)):
            if a[i] < a[i - 1]:
                a[i - 1], a[i] = a[i], a[i - 1]
                print(_, i, a[i], a[i - 1])


bubble_sort(a)

import numpy as np

np.linspace(0, 100, 26)

a = "pretzel"
i = 2
a[i] < a[i - 1]
"p" > "o"

a[0] > a[1] > a[2] < a[3]


adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]

for x in adj:
    for y in fruits:
        print(x, y)
