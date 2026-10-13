"""Compare a synthetic retrieval's posterior with truth.json, after labelling
the sectors cold/hot by temperature in both.

    python ../toi6894_smc/retrieve_toi6894_smc.py F01_s0.csv
    python check_recovery.py retrieval_result.pkl
"""
import json
import pickle
import sys

import numpy as np

from platon.terminator import label_by_temperature

result = pickle.load(open(sys.argv[1], "rb"))
names, samples = label_by_temperature(result.fit_info, result.equal_samples)
_, truth = label_by_temperature(result.fit_info, np.array(json.load(open("truth.json"))["values"]))
lo95, lo68, hi68, hi95 = np.percentile(samples, [2.5, 16, 84, 97.5], axis=0)
in68 = (lo68 <= truth[0]) & (truth[0] <= hi68)
in95 = (lo95 <= truth[0]) & (truth[0] <= hi95)
for i, name in enumerate(names):
    print(f"{name:22s} truth {truth[0, i]:11.4g}  68% [{lo68[i]:11.4g}, {hi68[i]:11.4g}]"
          + ("" if in68[i] else "  outside 68%" if in95[i] else "  outside 95%"))
print(f"truth inside 68%: {in68.sum()}/{len(names)}, inside 95%: {in95.sum()}/{len(names)}; "
      f"lnZ = {result.final_logz:.2f}")
