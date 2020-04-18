from random import Random

def mergeSort(ls : list):
    if len(ls) > 1:
        mid = len(ls) // 2
        l = ls[:mid]
        r = ls[mid:]
        mergeSort(l)
        mergeSort(r)
        i = j = k = 0
        while i < len(l) or j < len(r):
            if j == len(r) or i < len(l) and l[i] < r[j]:
                ls[k] = l[i]
                i += 1
            else:
                ls[k] = r[j]
                j += 1
            k += 1


if __name__ == "__main__":
    SEED = 0xF1
    a = [x for x in range(1,21)]
    rndEngine = Random(SEED)
    rndEngine.shuffle(a)
    print(a)
    mergeSort(a)
    print(a)
