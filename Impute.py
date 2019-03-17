import pandas as pd

def main(filepath):

    # Load files and create DataFrame
    df = pd.read_csv(filepath, sep=',', header=None)
    training = pd.read_csv('/Users/qiujingye/Downloads/credit 2019/crx.data.training', sep=',', header=None)
    testing = pd.read_csv('/Users/qiujingye/Downloads/credit 2019/crx.data.testing', sep=',', header=None)

    # Combine training file and testing file for better accuracy
    frames = [training, testing]
    result = pd.concat(frames)

    # For text features, replace "?" with modes
    num=[0, 3, 4, 5, 6, 8, 9, 11, 12]
    for i in num:
        df[i].loc[df[i] == '?'] = result[i].loc[result[i] != '?'].mode()[0]
        print('Replaced all "?" of feature ' + str(i+1) + ' with ' + str(result[i].loc[result[i] != '?'].mode()[0]))

    # Break the dataset into 2 parts ("+" and "-")
    df1 = result.loc[result[15] == '+']
    df2 = result.loc[result[15] == '-']

    # Change feature 2 and 14 to numerical
    df[1].loc[df[1]!='?'] = pd.to_numeric(df[1].loc[df[1]!='?'])
    df[13].loc[df[13] != '?'] = pd.to_numeric(df[13].loc[df[13] != '?'])

    # Replace "?" with mean for real-valued features
    df[1].loc[(df[1] == '?') & (df[15] == '+')] = pd.to_numeric(df1[1], errors='coerce').mean()
    df[2].loc[(df[2] == '?') & (df[15] == '+')] = df1[2].loc[df1[2]!= '?'].mean()
    df[7].loc[(df[7] == '?') & (df[15] == '+')] = df1[7].loc[df1[7]!= '?'].mean()
    df[10].loc[(df[10] == '?') & (df[15] == '+')] = df1[10].loc[df1[10]!= '?'].mean()
    df[13].loc[(df[13] == '?') & (df[15] == '+')] = pd.to_numeric(df1[13], errors='coerce').mean()
    df[14].loc[(df[14] == '?') & (df[15] == '+')] = df1[14].loc[df1[14]!= '?'].mean()
    # Ones with "-"
    df[1].loc[(df[1] == '?') & (df[15] == '-')] = pd.to_numeric(df2[1], errors='coerce').mean()
    df[2].loc[(df[2] == '?') & (df[15] == '-')] = df2[2].loc[df2[2]!= '?'].mean()
    df[7].loc[(df[7] == '?') & (df[15] == '-')] = df2[7].loc[df2[7]!= '?'].mean()
    df[10].loc[(df[10] == '?') & (df[15] == '-')] = df2[10].loc[df2[10]!= '?'].mean()
    df[13].loc[(df[13] == '?') & (df[15] == '-')] = pd.to_numeric(df2[13], errors='coerce').mean()
    df[14].loc[(df[14] == '?') & (df[15] == '-')] = df2[14].loc[df2[14]!= '?'].mean()

    # Print action
    print('Replaced all "?" in "+"labelled feature 2 with ' + str(pd.to_numeric(df1[1], errors='coerce').mean()))
    print('Replaced all "?" in "+"labelled feature 3 with ' + str(df1[2].loc[df1[2]!= '?'].mean()))
    print('Replaced all "?" in "+"labelled feature 8 with ' + str(df1[7].loc[df1[7]!= '?'].mean()))
    print('Replaced all "?" in "+"labelled feature 11 with ' + str(df1[10].loc[df1[10]!= '?'].mean()))
    print('Replaced all "?" in "+"labelled feature 14 with ' + str(pd.to_numeric(df1[13], errors='coerce').mean()))
    print('Replaced all "?" in "+"labelled feature 15 with ' + str(df1[14].loc[df1[14] != '?'].mean()))

    print('Replaced all "?" in "-"labelled feature 2 with ' + str(pd.to_numeric(df2[1], errors='coerce').mean()))
    print('Replaced all "?" in "-"labelled feature 3 with ' + str(df2[2].loc[df2[2] != '?'].mean()))
    print('Replaced all "?" in "-"labelled feature 8 with ' + str(df2[7].loc[df2[7] != '?'].mean()))
    print('Replaced all "?" in "-"labelled feature 11 with ' + str(df2[10].loc[df2[10] != '?'].mean()))
    print('Replaced all "?" in "-"labelled feature 14 with ' + str(pd.to_numeric(df2[13], errors='coerce').mean()))
    print('Replaced all "?" in "-"labelled feature 15 with ' + str(df2[14].loc[df2[14] != '?'].mean()))

    # Save file
    p = filepath + '.processed'
    df.to_csv(p,index=False,header=False)

    print('\nSuccess!\n\nProcessed data saved to '+p)

main('/Users/qiujingye/Downloads/credit 2019/crx.data.training')
main('/Users/qiujingye/Downloads/credit 2019/crx.data.testing')