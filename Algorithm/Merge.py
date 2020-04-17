def merge(x : list,y : list):
    ans = []
    i = 0
    j = 0
    while i < len(x) or j < len(y):
        if j == len(y) or i < len(x) and x[i] < y[j]:
            ans.append(x[i])
            i += 1
        else:
            ans.append(y[j])
            j += 1
    print(ans)
    return ans


if __name__ == "__main__":
    x = [1,2,3,4,5,6,7,8,9,10]
    y = [2,3,4,5,6]
    z = merge(x,y)
    for _ in z:
        print(_)