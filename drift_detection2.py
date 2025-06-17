import os
import pandas as pd
import ocpa.algo.filtering.log.time_filtering
import ocpa.algo.feature_extraction.factory as feature_extraction
from ocpa.algo.feature_extraction import time_series
from ocpa.objects.log.obj import OCEL
import numpy as np
from datetime import timedelta
import ruptures as rpt
from statsmodels.tsa.stattools import grangercausalitytests
import json

def avg(x):
    if len(x) == 0:
        return np.nan
    return sum(x)/len(x)

def process_file(filename):
    # Load data
    event_df = pd.read_csv(filename, sep=',')
    event_df["startTime"] = pd.to_datetime(event_df["startTime"])
    event_df["completeTime"] = pd.to_datetime(event_df["completeTime"])
    

    # Add required columns for OCEL
    event_df["event_id"] = list(range(0, len(event_df)))  # <- Add this line
    event_df["event_activity"] = event_df["event"]  # <- Add this line (optional but recommended)
    event_df["event_timestamp"] = event_df["startTime"]  # Map startTime to event_timestamp

    # Format case column as list of strings (critical fix!)
    event_df["case"] = event_df["case"].astype(str).map(lambda x: [f"case_{x}"])

    # Create OCEL object
    ocel = OCEL(event_df, ["case"])  # Assuming "case" is the object type
    
    # # Construct time series (example features)
    # s, time_index = time_series.construct_time_series(
    #     ocel,
    #     timedelta(days=7),
    #     [(avg, (feature_extraction.EVENT_SERVICE_TIME, ("startTime", "completeTime")))],
    #     [(sum, (feature_extraction.EXECUTION_IDENTITY,()))],
    #     ocpa.algo.filtering.log.time_filtering.events
    # )

    # Use simpler features that work with synthetic data
    s, time_index = time_series.construct_time_series(
        ocel,
        timedelta(hours=6),
        # Use EVENT_DURATION instead of SERVICE_TIME
        # With this corrected version:
        [(avg, (feature_extraction.EVENT_SERVICE_TIME, ("startTime", "completeTime")))],
        # Count cases instead of using IDENTITY
        [(sum, (feature_extraction.EXECUTION_IDENTITY, ()))],
        ocpa.algo.filtering.log.time_filtering.events
    )

    # Trim edges
    for k in s.keys():
        s[k] = s[k][4:-4]
    time_index = time_index[4:-4]

    # Add debug output after time series construction
    print("Time series shapes:")
    for k in s.keys():
        print(f"{k}: {s[k].shape}")

    loc = {}
    for k in s.keys():
        if len(s[k]) == 0:
            print(f"Warning: Empty time series for {k}")
            loc[k] = []
            continue
            
        # Only normalize if array not empty
        if np.max(s[k]) > 0:
            normalized = s[k]/np.max(s[k])
        else:
            normalized = s[k]  # Keep original if all zeros
        
        loc[k] = [bp for bp in rpt.Pelt().fit(normalized).predict(pen=0.1) 
              if bp != len(s[k])-1]

    
    # # Detect change points
    # loc = {
    #     k: [bp for bp in rpt.Pelt().fit(s[k]/np.max(s[k])).predict(pen=0.1) if bp != len(s[k])-1]
    #     for k in s.keys()
    # } 
    
    # Find Granger-causal relationships
    explainable_drifts = []
    p = 0.1
    
    for feat_1 in s.keys():
        for feat_2 in s.keys():
            if feat_1 == feat_2:
                continue
            loc_1 = loc[feat_1]
            loc_2 = loc[feat_2]
            for d in loc_1:
                for d_ in loc_2:
                    if d_ < d:
                        try:
                            res = grangercausalitytests(
                                pd.DataFrame({feat_1: s[feat_1], feat_2: s[feat_2]}), 
                                [d - d_]
                            )
                            p_value = res[d - d_][0]['ssr_ftest'][1]
                            if p_value <= p:
                                explainable_drifts.append((feat_1, feat_2, d, d_, p_value))
                        except:
                            continue
    
    # Save results
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output = {
        'time_index': [str(dt) for dt in time_index],
        'time_series': {str(k): v.tolist() for k, v in s.items()},
        'explainable_drifts': [
            (str(f1), str(f2), int(d), int(d_), float(p)) 
            for f1, f2, d, d_, p in explainable_drifts
        ]
    }
    
    with open(f'results/drift_results/{base_name}.json', 'w') as f:
        json.dump(output, f, indent=2)

def main():
    # Filter files matching criteria
    target_files = []
    for root, dirs, files in os.walk('synthetic_csv'):
        for file in files:
            if not file.endswith('.csv'):
                continue
            parts = file.split('_')
            if (
                parts[0] == 'sudden' and  # Drift type
                (parts[1] in ['time', 'trace']) and  # Perspective
                (parts[2].startswith('noise') and parts[2][5:] in ['', '0', '5', '10', '15', '20']) and  # Noise
                parts[3] == '100' and  # Case count
                parts[4] == 'cb.csv'  # Change pattern
            ):
                target_files.append(os.path.join(root, file))
    
    # Process each file
    for file in target_files:
        print(f"Processing {file}")
        process_file(file)

if __name__ == '__main__':
    main()