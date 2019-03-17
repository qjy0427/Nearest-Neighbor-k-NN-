import pandas as pd

def main():

    # Load files
    d = pd.read_csv('/Users/qiujingye/Downloads/credit 2019/Distance', sep=',', header=None)
    df0 = pd.read_csv('/Users/qiujingye/Downloads/credit 2019/crx.data.training.processed', sep=',', header=None)
    df1 = pd.read_csv('/Users/qiujingye/Downloads/credit 2019/crx.data.testing.processed', sep=',', header=None)

    # Calculate accuracy
    v = []
    for k in range(1, len(df1)):
        check = 0
        for i in range(len(df1)):
            num = d.nsmallest(k, i)
            temp = []
            for j in num.index.values:
                temp.append(df0[15][j])
            result = max(temp, key=temp.count)
            if result == df1[15][i]:
                check += 1
            accuracy = check / len(df1)
        v.append(accuracy)

        print('When k = ' + str(k) + ', the accuracy is ' + str(accuracy))
    v = pd.DataFrame(v)
    best = v[0].idxmax() + 1

    print('\nThe best k value is ' + str(best))

    # Save additional labelled data
    sign = []
    for i in range(len(df1)):
        num = d.nsmallest(best, i)
        temp = []
        for j in num.index.values:
            temp.append(df0[15][j])
        sign.append(max(temp, key=temp.count))
    df1[16] = sign

    df1.to_csv('/Users/qiujingye/Downloads/credit 2019/LabelledTesting', index=False, header=False)

    print('\nLabelled testing data saved.')

main()