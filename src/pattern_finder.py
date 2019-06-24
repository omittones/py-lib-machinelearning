from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import glob
import mass_ts as mts
import numpy as np
import random

def main():
    path = r'D:\Code\cs-lib-bitcoin-predictions\omittones\datasets\forex'
    all_files = glob.glob(path + "/*.csv")

    li = []
    for filename in all_files:
        print(f'Reading {filename}')
        df = pd.read_csv(filename, index_col=None, header=0, parse_dates=True)
        df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        li.append(df)
        if len(li) > 3:
            break
    frame = pd.concat(li, axis=0, ignore_index=False)
    print(f'Read {frame.size} rows.')

    prices = frame.set_index('Gmt time')
    prices = prices['High'].resample('1d').mean()
    resolution = timedelta(days=1)

    # temporary freeze date while debugging
    random_date = datetime(2007, 1, 1, 0, 0, 0)
    query = prices[random_date:random_date + resolution * 24]

    subset = prices
    distances = mts.mass2(subset.values, query.values)
    distances = np.absolute(distances)
    print(distances)

    # best_indices, best_dists = mts.mass2_batch(subset.values, query.values, batch_size=1000, top_matches=3)
    # print(best_indices)
    # print(best_dists)
    # for ax, idx in zip(axes[1:], best_indices):
    #     match = subset.iloc[idx : idx + len(query)]
    #     match.plot(ax = ax)
    #     ax.set_ylabel(f'Index {idx}')

    # # mass2_batch
    # # start a multi-threaded batch job with all cpu cores and give me the top 5 matches.
    # # note that batch_size partitions your time series into a subsequence similarity search.
    # # even for large time series in single threaded mode, this is much more memory efficient than
    # # MASS2 on its own.
    # batch_size = 10000
    # top_matches = 5
    # n_jobs = -1
    # indices, distances = mts.mass2_batch(ts, query, batch_size,
    #     top_matches=top_matches, n_jobs=n_jobs)

    # # find minimum distance
    # min_idx = np.argmin(distances)

    # find top 4 motif starting indices
    k = 4
    exclusion_zone = 25
    top_motifs = mts.top_k_motifs(distances, k, exclusion_zone)

    # # find top 4 discord starting indices
    # k = 4
    # exclusion_zone = 25
    # top_discords = mts.top_k_discords(distances, k, exclusion_zone)