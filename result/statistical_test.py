import csv
import os
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp, wilcoxon
from scipy.stats import t

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def bootstrap_confidence_interval(data, confidence=0.95, n_bootstrap=10000):
    medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, (1 - confidence) / 2 * 100)
    upper = np.percentile(medians, (1 + confidence) / 2 * 100)
    return np.median(data), lower, upper

def  test(data1, data2, alpha=0.05):
    # stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    stat, p_value = wilcoxon(data1, data2)

    # Output the results
    # print(f"Statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpretation
    if p_value < alpha:
        print("Reject the null hypothesis: The distributions are different.")
    else:
        print("Fail to reject the null hypothesis: The distributions are not significantly different.")


def pairwise_test_for_result():
    muFile = os.path.join(base_path, "result", "mu" + ".csv")
    staticFile = os.path.join(base_path, "result", "static" + ".csv")
    msFile = os.path.join(base_path, "result", "ms" + ".csv")
    scratchFile = os.path.join(base_path, "result", "scratch" + ".csv")
    scFile = os.path.join(base_path, "result", "sc" + ".csv")

    A=['Static', 'Scratch', 'MU', 'MS','SC']
    B={}
    static_acc = []
    with open(staticFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            static_acc.append(round(float(row[1].strip()) * 100.0, 2))
    B['Static']=static_acc

    ms_acc = []
    with open(msFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            ms_acc.append(round(float(row[1].strip()) * 100.0, 2))
    B['MS'] = ms_acc
    mu_acc = []
    with open(muFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            mu_acc.append(round(float(row[1].strip()) * 100.0, 2))
    B['MU'] = mu_acc
    scratch_acc = []
    with open(scratchFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            scratch_acc.append(round(float(row[1].strip()) * 100.0, 2))

    B['Scratch'] = scratch_acc
    sc_acc = []
    with open(scFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            sc_acc.append(round(float(row[1].strip()) * 100.0, 2))
    B['SC'] = sc_acc

    for i,a in enumerate(A):
        for j,b in enumerate(A):
            if j<=i:
                continue
            print(a+' vs. '+b)
            test(B[a], B[b], alpha=0.01)


def pairwise_test_for_scratch():
    scratchFile100 = os.path.join(base_path, "result", "mu" + ".csv")
    scratchFile75 = os.path.join(base_path, "result", "scratch_75%" + ".csv")
    scratchFile50 = os.path.join(base_path, "result", "scratch_50%" + ".csv")
    scratchFile25 = os.path.join(base_path, "result", "scratch_25%" + ".csv")

    A=['100', '75', '50', '25']
    B={}

    scratch_acc = []
    with open(scratchFile100) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            scratch_acc.append(round(float(row[1].strip()) * 100.0, 2))

    B['100'] = scratch_acc
    scratch_acc75 = []
    with open(scratchFile75) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            scratch_acc75.append(round(float(row[1].strip()) * 100.0, 2))

    B['75'] = scratch_acc75

    scratch_acc50 = []
    with open(scratchFile50) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            scratch_acc50.append(round(float(row[1].strip()) * 100.0, 2))

    B['50'] = scratch_acc50

    scratch_acc25 = []
    with open(scratchFile25) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for (serial, row) in enumerate(csv_reader):
            scratch_acc25.append(round(float(row[1].strip()) * 100.0, 2))

    B['25'] = scratch_acc25

    for i,a in enumerate(A):
        for j,b in enumerate(A):
            if j<=i:
                continue
            print(a+' vs. '+b)
            test(B[a], B[b], alpha=0.05)
            # print(bootstrap_confidence_interval(B[a]))
            # print(bootstrap_confidence_interval(B[b]))

            # print(confidence_interval(B[a]))
            # print(confidence_interval(B[b]))

# pairwise_test_for_result()
# pairwise_test_for_scratch()

test([55.33,52.56,56.37,50.80,48.47,50.91], [54.6,53.22,55.47,53.64,48.85,	51.39], alpha=0.05)
test([53.17,47.49,51.46,50.39,47.97,50.15], [54.6,53.22,55.47,53.64,48.85,	51.39], alpha=0.05)
test([52.69,55.65,52.21,47.05,55.132,56.518], [48.66,55.97866667,51.816,45.87133333,53.26933333,55.66666667], alpha=0.05)
test([52.69,55.65,52.21,47.05,55.132,56.518], [52.912,55.13533333,52.34333333,46.878,53.742,56.166], alpha=0.05)