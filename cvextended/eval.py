#!/usr/bin/env python3

"""Utility functions for evaluating nested cross-validation results"""

# Authors: Lyubomir Danov <->
# License: -

import pandas

def convert_to_DF(results):
    results_df = []
    for run_res in results:
        results_df.append(pandas.DataFrame(run_res))
    
    return pandas.concat(results_df)
