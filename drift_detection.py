# drift_detection.py
import pandas as pd
import ocpa.algo.feature_extraction.factory as feature_extraction
import ocpa.algo.filtering.log.time_filtering
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

def main():
    # Load data
    filename = "example_logs/mdl/BPI2017_new.csv"
    ots = ["application", "offer"]
    
    event_df = pd.read_csv(filename, sep=',')
    event_df["event_timestamp"] = pd.to_datetime(event_df["event_timestamp"])
    
    # Process object types
    for ot in ots:
        event_df[ot] = event_df[ot].map(
            lambda x: [y.strip() for y in x.split(',')] if isinstance(x, str) else [])
    
    # Create OCEL object
    event_df["event_id"] = list(range(0, len(event_df)))
    event_df.index = list(range(0, len(event_df)))
    event_df["event_start_timestamp"] = pd.to_datetime(event_df["event_start_timestamp"])
    ocel = OCEL(event_df, ots)
    
    # Construct time series for example features
    s, time_index = time_series.construct_time_series(
        ocel,
        timedelta(days=7),
        [(avg, (feature_extraction.EVENT_SERVICE_TIME, ("event_start_timestamp", "W_Validate application")))],
        [(sum, (feature_extraction.EXECUTION_IDENTITY,()))],
        ocpa.algo.filtering.log.time_filtering.events
    )
    
    # Trim edges
    for k in s.keys():
        s[k] = s[k][4:-4]
    time_index = time_index[4:-4]
    
    # Detect change points
    loc = {
        k: [bp for bp in rpt.Pelt().fit(s[k]/np.max(s[k])).predict(pen=0.1) if bp != len(s[k])-1] 
        for k in s.keys()
    }
    
    # Find Granger-causal relationships
    explainable_drifts = []
    p = 0.05
    
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
    output = {
        'time_index': [str(dt) for dt in time_index],
        'time_series': {str(k): v.tolist() for k, v in s.items()},
        'explainable_drifts': [
            (str(f1), str(f2), int(d), int(d_), float(p)) 
            for f1, f2, d, d_, p in explainable_drifts
        ]
    }
    
    with open('drift_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Found {len(explainable_drifts)} explainable drifts")
    return output

if __name__ == '__main__':
    main()