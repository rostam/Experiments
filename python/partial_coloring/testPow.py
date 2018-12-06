for a in range(1, 100):
    t1 = 2**(a/10)
    t2 = 2 ** ((a + 1) / 10) - 2 ** (a / 10)
    print(t1)
    print(t2)
    print(t2/t1)
    print(2**0.1 - 1)
    print(2**0.1 - 1 - t2/t1)
    print("\n")
