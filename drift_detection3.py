import os
import pandas as pd
import ocpa.algo.filtering.log.time_filtering
import ocpa.algo.feature_extraction.factory as feature_extraction
from ocpa.algo.feature_extraction import time_series
from ocpa.objects.log.obj import OCEL
import numpy as np
from datetime import timedelta
import ruptures as rpt
import json

def avg(x):
    if len(x) == 0:
        return np.nan
    return sum(x)/len(x)

def validate_time_series(s):
    """Ensure time series contains valid values"""
    validated = {}
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        # Replace NaN values
        v = [x if x is not None and not np.isnan(x) else 0 for x in v]
        validated[k] = np.array(v)
    return validated

def process_file(filename):
    try:
        # Load data
        event_df = pd.read_csv(filename, sep=',')
        event_df["startTime"] = pd.to_datetime(event_df["startTime"])
        event_df["completeTime"] = pd.to_datetime(event_df["completeTime"])
        
        # Add required columns for OCEL
        event_df["event_id"] = list(range(0, len(event_df)))
        event_df["event_activity"] = event_df["event"]
        event_df["event_timestamp"] = event_df["startTime"]
        
        # Format case column as list of strings
        event_df["case"] = event_df["case"].astype(str).map(lambda x: [f"case_{x}"])
        
        # Create OCEL object
        ocel = OCEL(event_df, ["case"])
        
        # Define features compatible with synthetic data
        feat_events = [
            (avg, (feature_extraction.EVENT_SERVICE_TIME, ("startTime", "completeTime"))),
            (avg, (feature_extraction.EVENT_NUM_OF_OBJECTS, ()))
        ]
        
        feat_executions = [
            (sum, (feature_extraction.EXECUTION_IDENTITY, ())),
            (avg, (feature_extraction.EXECUTION_NUM_OF_EVENTS, ()))
        ]
        
        # Construct time series
        try:
            s, time_index = time_series.construct_time_series(
                ocel,
                timedelta(hours=6),
                feat_events,
                feat_executions,
                ocpa.algo.filtering.log.time_filtering.events
            )
        except Exception as e:
            print(f"Time series construction failed: {str(e)}")
            return
        
        # Validate and trim time series
        s = validate_time_series(s)
        for k in s.keys():
            s[k] = s[k][1:-1]
        time_index = time_index[1:-1] if len(time_index) > 8 else time_index
        
        # Detect change points
        loc = {}
        for k in s.keys():
            if len(s[k]) < 2:
                loc[k] = []
                continue
                
            try:
                normalized = s[k]/np.max(s[k]) if np.max(s[k]) > 0 else s[k]
                loc[k] = [bp for bp in rpt.Pelt().fit(normalized).predict(pen=0.05) 
                          if bp != len(s[k])-1]
            except Exception as e:
                print(f"Drift detection error for {k}: {str(e)}")
                loc[k] = []
        
        # Collect all detected drifts
        drifts = []
        for feat in s.keys():
            for bp in loc[feat]:
                drifts.append((str(feat), int(bp)))
        
        # Save results
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output = {
            'time_index': [str(dt) for dt in time_index],
            'time_series': {str(k): v.tolist() for k, v in s.items()},
            'drifts': drifts
        }
        
        # Ensure output directory exists
        os.makedirs('results/drift_results', exist_ok=True)
        
        with open(f'results/drift_results/{base_name}.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        return output
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

def main():
    # Define all valid patterns
    DRIFT_TYPES = ['gradual', 'incremental', 'recurring', 'sudden']
    PERSPECTIVES = ['time', 'trace']
    NOISE_LEVELS = ['0', '5', '10', '15', '20']
    CASE_COUNTS = ['100', '500', '1000']
    CHANGE_PATTERNS = ['cb']
    # , 'cd', 'cf', 'cp', 'IOR', 'IRO', 'lp', 'OIR', 'pl', 
    #                   'pm', 're', 'RIO', 'ROI', 'rp', 'sw']
    
    # Filter files matching criteria
    target_files = []
    for root, dirs, files in os.walk('synthetic_csv'):
        for file in files:
            if not file.endswith('.csv'):
                continue
                
            # Split filename components
            parts = file.split('_')
            if len(parts) < 5:  # Skip malformed filenames
                continue
                
            drift_type = parts[0]
            perspective = parts[1]
            noise_part = parts[2]
            case_count = parts[3]
            pattern_part = parts[4]
            
            # Extract noise level from "noiseX" pattern
            if not noise_part.startswith('noise'):
                continue
            noise_level = noise_part[5:] or '0'  # Handle "noise0" case
            
            # Extract pattern without .csv extension
            pattern = pattern_part.split('.')[0]
            
            # Validate against all criteria
            if (drift_type in DRIFT_TYPES and 
                perspective in PERSPECTIVES and 
                noise_level in NOISE_LEVELS and 
                case_count in CASE_COUNTS and 
                pattern in CHANGE_PATTERNS):
                
                target_files.append(os.path.join(root, file))
    
    print(f"Found {len(target_files)} files to process")
    
    # Process each file
    for file in target_files:
        print(f"Processing {file}")
        process_file(file)

if __name__ == '__main__':
    main()