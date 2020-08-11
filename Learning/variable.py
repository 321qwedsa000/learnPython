def main():
    varInt = 10
    varDouble = 10.0
    varBool = True
    varStr = "string"
    print("{varInt}\t{varDouble}\t{varBool}\t{varStr}".format
    (
        varInt=varInt,
        varDouble=varDouble,
        varBool=varBool,
        varStr=varStr
    ))

if __name__ == "__main__":
    main()