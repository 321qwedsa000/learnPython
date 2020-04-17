from Merge import merge

class MergeSortBase:
    def m(self,ls: list,f:int,m:int,l:int):
        left = ls[f:m]
        right = ls[m:]
        
    @staticmethod
    def mergeSort(ls : list, f: int, l: int):
        if f < l:
            o = MergeSortBase()
            mid = (f+l)//2
            MergeSortBase.mergeSort(ls,f,mid)
            MergeSortBase.mergeSort(ls,mid+1,l)
            o.m(ls,f,mid,l)

def mergeSort(ls : list):
    first = 0
    last = len(ls)-1
    MergeSortBase.mergeSort(ls,first,last)


if __name__ == "__main__":
    a = [1,4,5,3,2,7,6,8]
    mergeSort(a)
    print(a)