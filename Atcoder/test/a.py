x = int(input())
a = {}
for i in range(x):
    print(i)
a[1] = 2
if a.get(2) == None:
    print("not exist")
