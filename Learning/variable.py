def main() -> None:
    varInt = 10
    varFloat = 10.0
    varBool = True 
    varStr = "string"
    print("varInt     varFloat     varBool     varStr")
    print("{varInt}     {varFloat}     {varBool}     {varStr}".format
    (
        varInt=varInt,
        varFloat=varFloat,
        varBool=varBool,
        varStr=varStr
    ))

if __name__ == "__main__":
    main()