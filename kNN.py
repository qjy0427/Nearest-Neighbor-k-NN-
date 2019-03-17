import pandas as pd
from math import sqrt

def main():
    # Load files
    df0 = pd.read_csv('/Users/qiujingye/Downloads/credit 2019/crx.data.training.processed', sep=',', header=None)
    df1 = pd.read_csv('/Users/qiujingye/Downloads/credit 2019/crx.data.testing.processed', sep=',', header=None)
    # Combine for better accuracy
    frames = [df0, df1]
    result = pd.concat(frames)
    std = result.std()      # Standard deviation for z-scaling

    d = pd.DataFrame(index = [], columns = range(len(df1)))     # Create distance DataFrame

    # Sort
    values = [1, 2, 7, 10, 13, 14]
    strings = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    # Calculate distance
    for i in range(len(df1)):
        temp = []
        for k in range(len(df0)):
            terms = 0
            for j in values:
                terms += ((df0[j][k] - df1[j][i])/std.loc[j]) ** 2
            for l in strings:
                if df0[l][k] != df1[l][i]:
                    terms += 1
            temp.append(sqrt(terms))
        d[i] =  temp
        print('\r' + str(i+1) + '/' + str(len(df1)) + ' finished',end='')
    # Save distance data
    d.to_csv('/Users/qiujingye/Downloads/credit 2019/Distance', index=False, header=False)

    print('\n\nDistance data saved.\n')

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
        num = d.nsmallest(k, i)
        temp = []
        for j in num.index.values:
            temp.append(df0[15][j])
        sign.append(max(temp, key=temp.count))
    df1[16] = sign

    df1.to_csv('/Users/qiujingye/Downloads/credit 2019/LabelledTesting', index=False, header=False)

    print('\nLabelled testing data saved.')

main()