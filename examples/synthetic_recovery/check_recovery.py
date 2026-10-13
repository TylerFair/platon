"""Compare a synthetic retrieval's posterior with the truth.

    python check_recovery.py F01_s0_results/retrieval_result_F01_s0.pkl F01_s0_truth.json

Sectors are labelled cold/hot by temperature in both the samples and the
truth before comparing.  Prints each parameter's posterior interval and
whether the truth lies inside the central 68% and 95% intervals.
"""
import argparse
import json
import pickle

import numpy as np

from platon.terminator import label_by_temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("truth")
    args = parser.parse_args()

    with open(args.result, "rb") as f:
        result = pickle.load(f)
    truth = json.load(open(args.truth))
    fit_info = result.fit_info
    if list(fit_info.fit_param_names) != truth["names"]:
        raise ValueError("result and truth have different parameters")
    names, samples = label_by_temperature(fit_info, result.equal_samples)
    _, true = label_by_temperature(fit_info, np.array(truth["values"]))

    in68 = in95 = 0
    print(f"{'parameter':22s} {'truth':>11s} {'2.5%':>11s} {'16%':>11s} {'50%':>11s} {'84%':>11s} {'97.5%':>11s}")
    for i, name in enumerate(names):
        q = np.percentile(samples[:, i], [2.5, 16, 50, 84, 97.5])
        t = true[0, i]
        a, b = q[1] <= t <= q[3], q[0] <= t <= q[4]
        in68 += a
        in95 += b
        flag = "" if a else (" (outside 68%)" if b else " (outside 95%)")
        print(f"{name:22s} {t:11.4g} " + " ".join(f"{v:11.4g}" for v in q) + flag)
    n = len(names)
    print(f"truth inside 68%: {in68}/{n}, inside 95%: {in95}/{n}")
    print(f"lnL(truth) = {truth['lnl_truth']:.2f}; lnZ = {result.final_logz:.2f}")


if __name__ == "__main__":
    main()
