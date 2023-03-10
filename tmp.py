
len_train = 15
eval_now = 100

print(list(map(lambda x: ((x - 1) // len_train) + 1, range(100, 15 * 501, 100))))

index = 0
for epoch in range(1, 50 + 1):
    for i in range(len_train):
        print(index, epoch, i)
        index += 1
        if epoch in map(lambda x: ((x - 1) // len_train) + 1, range(100, 15 * 501, 100)):
            print(True)
        else:
            print(False)
        if (index + 1) % eval_now == 0 and i > 0:
            print('eval!!!!!!')
    print()