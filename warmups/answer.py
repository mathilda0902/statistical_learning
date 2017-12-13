'''closest([10, 17, 2, 29, 16], 14) return 16'''
import numpy as np

def closest(lst, num):
    arr = np.array(lst)
    return arr[np.argmin(abs(arr - num))]

'''Given an infinite number of US coins (coins = [1, 5, 10, 25]) and an amount
value in cents, what are minimum number of coins needed to make change for value?
Write a function find_change that takes as input the coin denominations coins,
and value as the amount in cents. Your function should return the minimum amount
of coins necessary to make change for the specified value as an integer.

In [23]: find_change(coins, 100)
4
In [24]: find_change(coins, 74)
8'''

coins = [1, 5, 10, 25]
def find_change(coins, amount):
    pocket = []
    for c in coins[::-1]:
        x, y = np.modf(amount / float(c))
        pocket.append(y)
        amount -= y * c
    return int(sum(pocket))
