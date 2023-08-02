import pandas as pd
from matplotlib import pyplot as plt
import datetime
import affinegap
import re
from functools import lru_cache
import random
import Levenshtein

NUM_CARDS = 30
IGNORE_CHARS = ['.', '-', '(', ')']
IGNORE_LIST_PATH = "ignore_list.txt"
TEST_SET_PATH = "Metainventory_Version1.0.0.csv"
TEST_SAMPLE_PATH = 'data/disease_sample_gt.txt'
DISEASE_SET_PATH = "data/disease_names_gt.txt"
AREA_SET_PATH = "data/area_names_sample_gt.txt"
POLICE_SET_PATH_SMALL = "data/titles_pruned.txt"
POLICE_SET_PATH_FULL = "data/titles.txt"

timing_list = [0]*5

mmw = 11
cutoff = 10

# distance metric based on dynamic programming algorithm
def dp_distance(s1, s2, short=True, ignore=True):
    # handle cases and ordering
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    # heuristic for skipping the dp algorithm for speedup
    if s1[0] != s2[0]:
        return 1000

    # parse longer word into array, including spaces
    words_arr = [word + ' ' for word in s1.split(' ')]
    words_arr[-1] = words_arr[-1][:-1]

    # cache first letter of words
    fls = [words_arr[i][0] for i in range(len(words_arr))]

    # timing_list[2] += (datetime.datetime.now()-t).total_seconds()
    # returns the distance between string 1 up to the kth word and string 2 up to the pth character
    @lru_cache(maxsize=None)
    def dp(k, p):
        if k == 0 and p == 0:
            # when out of words in string 1 we want to be out of characters in string 2
            return 0
        if k == 0 or p == 0:
            # return infinity, or something close to it
            return 1000
        word, rest = words_arr[k-1], s2[:p]
        substr = ""

        # timing_list[3] += (datetime.datetime.now()-t).total_seconds()
        # handle words like "in" or "the" which may be skipped in abbreviations: 
        # allow them to be skipped
        # or check if word is in ignore list
        if (short and len(word) < 5) or (ignore and word in ignore_list):
            min_dist = dp(k-1, p)
        else:
            min_dist = float('inf')
        # print(words_arr)
        while p > 0:
        # while p > max(0, len(rest) - len(word) - 1):
            p -= 1
            if rest[p] in IGNORE_CHARS:
                continue
            substr = rest[p] + substr
            if word[0] != substr[0]:
                # return infinity, or something close to it
                value = 1000
            elif is_substring(word, substr):
                value = 0
            else:
                value = affinegap.normalizedAffineGapDistance(word, substr)
            if k-1 >= 0 and fls[k-1] != rest[p]:
                continue
            min_dist = min(min_dist, value + dp(k-1, p))
        return min_dist

    return dp(len(words_arr), len(s2))

def normalized_dp_distance(s1, s2):
    d = dp_distance(s1, s2)
    return 1-0.965981*0.887683**d

def normalized_dp_distance_2(s1, s2):
    return min(1, dp_distance(s1, s2)/cutoff)

def normalized_dp_distance_short_only(s1, s2):
    return min(1, dp_distance(s1, s2, short=True, ignore=False)/cutoff)

def normalized_dp_distance_ignore_only(s1, s2):
    return min(1, dp_distance(s1, s2, short=False, ignore=True)/cutoff)

def normalized_dp_distance_none(s1, s2):
    return min(1, dp_distance(s1, s2, short=False, ignore=False)/cutoff)

def normalized_dp_distance_3(s1, s2):
    d = dp_distance(s1, s2)
    normalizer = max(len(s1), len(s2))*mmw
    if d > normalizer:
        return 1
    return d/normalizer

def normalized_affine_gap(s1, s2):
    dist = affinegap.affineGapDistance(s1, s2, mismatchWeight = mmw)
    normalizer = max(len(s1), len(s2))*mmw
    return dist/normalizer

# edit distance metric
def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)/max(len(s1), len(s2))

def levenshtein_distance_2(s1, s2):
    return min(1, Levenshtein.distance(s1, s2)/cutoff)

def normalized_affine_gap_2(s1, s2):
    dist = affinegap.normalizedAffineGapDistance(s1, s2)
    return min(1, dist/cutoff)

#jaccard distance of two strings
def jaccard(str1, str2):
    n = 1
    if len(str1) < n and len(str2) < n:
        return 1
    s1, s2 = set([str1[i:i+n] for i in range(len(str1)-n+1)]), set([str2[i:i+n] for i in range(len(str2)-n+1)])
    return 1 - (len(s1 & s2) / len(s1 | s2))

def jaccard_word(str1, str2):
    s1, s2 = set(re.split('\W+', str1)), set(re.split('\W+', str2))
    return 1 - (len(s1 & s2) / len(s1 | s2))

def is_substring(str1, str2):
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    i = j = 0
    while(i < len(str1) and j < len(str2)):
        if str1[i] == str2[j]:
            j+=1
        i+=1
    return j == len(str2)

def count_captures(metric, limit):
    df = pd.read_csv(TEST_SET_PATH, sep="|").sample(500)
    df2 = df.copy()
    df2['LF'] = df['SF']
    df2['SF'] = df['LF']
    titles = set([title for title in pd.concat([df, df2])["SF"] if isinstance(title, str)])
    mat = {}
    for title in titles:
        mat[title] = {}
    for t1 in titles:
        for t2 in titles:
            mat[t1][t2] = mat[t2][t1] = metric(t1, t2)

    print('scoring')
    count = 0
    for i, row in df.iterrows():
        if not isinstance(row['SF'], str):
            continue
        correct = row['LF']
        sorted_lfs = list(df['LF'])
        sorted_lfs.sort(key=lambda lf: mat[row['SF']][lf])
        sorted_lfs = sorted_lfs[:limit]
        if correct in sorted_lfs:
            count += 1
    return count

def plot_count_captures(metrics, metric_names):
    trials = 5
    plt.title("Capture Counts for Metrics")
    plt.xlabel("k, size of set of closest words")
    plt.ylabel("Average Count")
    for i, metric in enumerate(metrics):
        x, y = [], []
        for k in range(1, 11):
            print(metric_names[i] + ' ' + str(k))
            avg_count = 0
            for _ in range(0, trials):
                avg_count += count_captures(metric, k)
            x.append(k)
            y.append(avg_count / trials)
        plt.plot(x, y, label=metric_names[i])
    print(timing_list)
    plt.xticks(x)
    plt.legend()
    plt.show()

def plot_time(sample_sizes, metrics, metric_names):
    y_list = [[] for _ in range(len(metrics))]
    for s in sample_sizes:
        df = pd.read_csv(TEST_SET_PATH, sep="|").sample(s)
        df2 = df.copy()
        df2['LF'] = df['SF']
        df2['SF'] = df['LF']
        titles = set([title for title in pd.concat([df, df2])["SF"] if isinstance(title, str)])
        mat = {}
        for title in titles:
            mat[title] = {}

        for i, metric in enumerate(metrics):
            t = datetime.datetime.now()
            for t1 in titles:
                for t2 in titles:
                    mat[t1][t2] = mat[t2][t1] = metric(t1, t2)
            y_list[i].append((datetime.datetime.now()-t).total_seconds())
    print(y_list)
    for i in range(len(y_list)):
        plt.plot(sample_sizes, y_list[i], label=metric_names[i])
    plt.title("Metric Timing")
    plt.xlabel("Number of rows sampled from dataset")
    plt.ylabel("Runtime in seconds")
    plt.legend()
    plt.show()

def plot_time_word_length(metrics, metric_names):
    summer = {}
    counter = {}
    for name in metric_names:
        summer[name] = {}
        counter[name] = {}
    for _ in range(100):
        df = pd.read_csv(TEST_SET_PATH, sep="|").sample(500)
        df2 = df.copy()
        df2['LF'] = df['SF']
        df2['SF'] = df['LF']
        titles = set([title for title in pd.concat([df, df2])["SF"] if isinstance(title, str)])
        mat = {}
        for title in titles:
            mat[title] = {}

        for i, metric in enumerate(metrics):
            for t1 in titles:
                for t2 in titles:
                    t = datetime.datetime.now()
                    mat[t1][t2] = mat[t2][t1] = metric(t1, t2)
                    t = (datetime.datetime.now()-t).total_seconds()*1000
                    x = max(len(t1), len(t2))
                    if x in summer[metric_names[i]]:
                        summer[metric_names[i]][x] += t
                        counter[metric_names[i]][x] += 1
                    else:
                        summer[metric_names[i]][x] = t
                        counter[metric_names[i]][x] = 1
    for name in metric_names:
        x, y = [], []
        keys = list(summer[name].keys())
        keys.sort()
        for k in keys:
            x.append(k)
            y.append(summer[name][k]/counter[name][k])
        plt.plot(x, y, label=name)
    plt.title("Metric Timing Across Word Length")
    plt.xlabel("Length of longform word in characters")
    plt.ylabel("Average runtime in milliseconds")
    plt.legend()
    plt.show()

def calc_prec_recall(metric, df):
    mat = {}
    for sf in df['SF']:
        for lf in df['LF']:
            if not isinstance(sf, str):
                continue
            score = metric(lf, sf)
            if sf not in mat or score < mat[sf][0]:
                mat[sf] = (score, set([lf]))
            elif score == mat[sf][0]:
                closest = mat[sf][1]
                closest.add(lf)
                mat[sf] = (score, closest)
    print('scoring')
    return score_distance_mat(df, mat)

def calc_prec_recall_thresh(metric, df, thresh, df2=None):
    # df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    mat = {}
    for sf in df['SF']:
        for lf in df['LF']:
            if not isinstance(sf, str):
                continue
            score = metric(lf, sf)
            if sf not in mat:
                mat[sf] = set([])
            if score < thresh:
                closest = mat[sf]
                closest.add(lf)
                mat[sf] = closest
    print('scoring')
    if df2 is not None:
        return score_distance_mat(df2, mat)
    return score_distance_mat(df, mat)

def calc_prf_gpt_output(df, df2, thresh):
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    mat = {}
    for _, row in df2.iterrows():
        sf = row['SF']
        for i in range(df2.shape[1]//2):
            if sf not in mat:
                mat[sf] = set([])
            if 1 - row['PROB' + str(i+1)] < thresh:
                closest = mat[sf]
                closest.add(row['LF' + str(i+1)])
                mat[sf] = closest
    # print('scoring')
    return score_distance_mat(df, mat)

def score_distance_mat(df, mat):
    precision, recall, cnt = 0, 0, 0
    for _, row in df.iterrows():
        tp, fp, fn = 0, 0, 0
        if not isinstance(row['SF'], str):
            continue
        if row['SF'] == row['LF']:
            continue
        cnt += 1
        closest = mat[row['SF']]
        if row['LF'] in closest:
            tp += 1
            # account for same sf words appearing in set
            lfs = set(df.loc[df['SF'] == row['SF']]['LF'])
            fp += len(closest) - len(closest & lfs)
        else:
            fn += 1
            # print(row['SF'])
            # print(row['LF'])
            # print('======')
        if tp != 0:
            precision, recall = precision + tp/(tp+fp), recall + tp/(tp+fn)
        else:
            precision += 1
    precision, recall = precision/cnt, recall/cnt
    return precision, recall, 2*precision*recall/(precision+recall)

def get_df_from_pkduck_set(path):
    d = {}
    d['SF'] = []
    d['LF'] = []
    with open(path) as f:
        lines = f.readlines()
        sf = True
        for line in lines:
            if line == '\n':
                sf = True
                continue
            s = line[:-1]
            # old_s = s
            # s_arr = s.split(" ")
            # for word in ignore_list:
            #     s_arr = [i for i in s_arr if i != word]
            # s = " ".join(s_arr)
            # if old_s != s:
            #     print(s, old_s)
            if sf:
                d['SF'].append(s)
                sf = False
            else:
                d['LF'].append(s)
    return pd.DataFrame(d)

if __name__ == "__main__":
    # open ignore list
    file = open(IGNORE_LIST_PATH)
    ignore_list = set()
    for line in file:
        ignore_list.add(line[:-1])
    file.close()
    # print(count_captures(affinegap.normalizedAffineGapDistance, 5))
    # print(dp_distance("Sargeant", "sgt"))
    # print(dp_distance("Sargeant", "sgr"))
    # print(dp_distance("prtl off", "patrol officer"))
    # print(dp_distance("Senior Investigator", "sgr"))
    # print(dp_distance("Senior Investigator", "sir"))
    # print(dp_distance("dtpy", "deputy"))
    # print(dp_distance("Motor Carrier Inspector III", "Motor Carrier Inspector I"))
    # print(dp_distance("mci3", "maj"))
    # print(dp_distance("Assistant Park Manager", "apkmgr"))
    # print(dp_distance("Assistant Park Manager", "aojkrpkmgr"))
    # print(dp_distance("Sargeant", "Lieutenant"))
    # print(dp_distance("Master Trooper", "Master Deputy"))
    # print(dp_distance("Records", "Rngr"))
    # print(dp_distance("Property", "Spty"))
    # print(dp_distance("Marshall", "Maj"))
    # print(dp_distance("Mjr", "Maj"))
    # print(dp_distance("Major", "Maj"))
    # print(dp_distance("Patrol Officer II", "Patrol Officer IV"))
    # print(dp_distance("Administrative Aide 3P", "Administrative Aide 2P"))
    # print(dp_distance("Administrative 3P Aide", "Administrative 2P Aide"))
    # print(dp_distance("Deputy VI", "Deputy XI"))
    # print(dp_distance("sjx", "bpl"))
    # print(dp_distance("127th Precinct", "127PCT"))
    # print(dp_distance("Audits & Accounts Section", "AUD&ACC"))
    # print(dp_distance('A.C.', 'alternating current'))
    # print(dp_distance('123 DET', '123 Detective Squad'))
    # print(dp_distance('123 DET', '122 DET'))
    # print(dp_distance("west hwy patrol", "w highway ptl"))
    # print(dp_distance("mopett", "Moderate Pulmonary Embolism Treated with Thrombolysis"))
    # plot_time_word_length([dp_distance, Levenshtein.distance, affinegap.normalizedAffineGapDistance, jaccard], ["Smash", "Levenshtein", "Affine Gap", "Jaccard"])
    # plot_count_captures([dp_distance, Levenshtein.distance, affinegap.normalizedAffineGapDistance, jaccard], ["Smash", "Levenshtein", "Affine Gap", "Jaccard"])
    # plot_time([10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000], [jaccard_word], ["Jaccard Word"])
    
    # p,r,f = 0,0,0
    # trials = 10
    # for _ in range(0,trials):
    #     df = pd.read_csv(TEST_SET_PATH, sep="|").sample(684)
    #     pnew,rnew,fnew = calc_prec_recall_thresh(dp_distance, df, 1)
    #     p,r,f = p+pnew,r+rnew,f+fnew
    # print(p/trials,r/trials,f/trials)

    df2 = get_df_from_pkduck_set(POLICE_SET_PATH_SMALL)
    metrics = [normalized_dp_distance_short_only, normalized_dp_distance_ignore_only, normalized_dp_distance_none]

    for i in range(7, 10):
        print(i/10)
        print('===========')
        for m in metrics:
            print(m)
            df = get_df_from_pkduck_set(TEST_SAMPLE_PATH)
            print(calc_prec_recall_thresh(m, df, 1-i/10))

    # df = get_df_from_pkduck_set(POLICE_SET_PATH_FULL)
    # df2 = df.copy()
    # df2['LF'] = df['SF']
    # df2['SF'] = df['LF']
    # df = pd.concat([df, df2])
    # df['lengths'] = df['SF'].str.len()
    # print(df['lengths'].mean())
    # print(df['lengths'].std())
    # print(df['lengths'].median())


    # l = []
    # for t in range(1, 10):
    #     p,r,f = calc_prec_recall_thresh(levenshtein_distance, df, t/10)
    #     l.append((p, r))
    # print(l)

    # print(affinegap.affineGapDistance("mc", "meeeeeee",
    #                                   matchWeight = 1,
    #                              mismatchWeight = 11,
    #                              gapWeight = 10,
    #                              spaceWeight = 7,
    #                              abbreviation_scale = 1))
        

    # print(calc_prec_recall_thresh(Levenshtein.distance, df, 14))
    # print(calc_prec_recall_thresh(affinegap.normalizedAffineGapDistance, df, 1.5))
    # print(calc_prec_recall_thresh(jaccard_word, df, 0.3))