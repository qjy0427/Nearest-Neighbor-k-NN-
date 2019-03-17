import pandas as pd
import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt

def main():
    # Load ".mat" file, and create a DataFrame
    df = si.loadmat('/Users/qiujingye/Downloads/Homework2/detroit.mat')['data']
    df = pd.DataFrame(df)

    # Prepare the design matrix, combine FTP, WE and a test variable
    m = []
    for j in range(1, 8):
        temp = []
        for i in range(len(df)):
            row = [df[j][i],df[0][i],df[8][i]]
            temp.append(row)
        m.append(temp)

    # Test the rest 7 variables, and let t be the HOM column
    t = df[9]
    errors = []
    for i in range(7):
        # Create the design matrix p
        p = np.array(m[i])
        pb = range(3)
        tb = 0
        for j in range(len(p)):
            pb += p[j]
            tb += t[j]
        # Get the mean of Φ and t
        pb /= len(t)
        tb /= len(t)
        # Formula 1 on my report
        weights = (np.linalg.inv(p.T.dot(p))).dot(p.T).dot(t)
        # Formula 2 on my report
        wpb = 0
        for j in range(3):
            wpb += weights[j] * pb[j]
        w0 = tb - wpb
        # Formula 3 on my report
        e = 0
        for n in range(len(p)):
            wp = weights.T.dot(p[n])
            e += (t[n] - w0 - wp) ** 2
        e /= 2
        errors.append(e)

    print('\nThe errors of UEMP, MAN, LIC, GR, NMAN, GOV, HE are listed below:')
    print(errors)
    print('(The one with the smallest error is the third variable)')
    print('And the plot is as shown.')
    # Plot
    plt.plot(range(1, 8), errors, linestyle = '-', color = 'green', marker = 'o')
    plt.ylabel('Error')
    plt.show()

main()