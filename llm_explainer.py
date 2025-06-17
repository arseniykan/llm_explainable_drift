
# llm_explainer.py
import json
import openai
import anthropic
import tiktoken
import numpy as np
import logging
import os
from datetime import datetime
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='llm_explainer.log'
)
logger = logging.getLogger(__name__)

class DriftExplainer:
    def __init__(self, max_tokens=512, temperature=0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise

    def _truncate_prompt(self, prompt, max_tokens=3072):
        try:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > max_tokens:
                logger.info(f"Prompt truncated from {len(tokens)} tokens")
                return self.tokenizer.decode(tokens[:max_tokens]) + "... [TRUNCATED]"
            return prompt
        except Exception as e:
            logger.error(f"Prompt truncation error: {str(e)}")
            return prompt

    def explain_drift(self, model, feature_pair, drift_data):
        try:
            prompt = self._build_prompt(feature_pair, drift_data)
            prompt = self._truncate_prompt(prompt)
            
            if model.startswith("gpt"):
                return self._call_openai(model, prompt)
            elif model.startswith("claude"):
                return self._call_claude(model, prompt)
            raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            logger.error(f"Drift explanation error: {str(e)}")
            return f"Explanation failed: {str(e)}"

    def _build_prompt(self, feature_pair, drift_data):
        try:
            feat_a, feat_b = feature_pair
            drift_point = drift_data['drift_point']
            print(f"""Analyze this concept drift relationship between features:
            
            Feature A: {feat_a}
            Feature B: {feat_b}
            P-value: {drift_data['p_value']:.4f}
            Drift location: {drift_point['timestamp']}
            
            Time Series Values Around Drift:
            Feature A (before): {drift_data['values']['a_before']}
            Feature A (after): {drift_data['values']['a_after']}
            Feature B (before): {drift_data['values']['b_before']}
            Feature B (after): {drift_data['values']['b_after']}
            
            Please explain:
            1. What business process change this drift might indicate
            2. Potential causal relationships between these features
            3. Recommended actions for process improvement
            """)
            return f"""Analyze this concept drift relationship between features:
            
            Feature A: {feat_a}
            Feature B: {feat_b}
            P-value: {drift_data['p_value']:.4f}
            Drift location: {drift_point['timestamp']}
            
            Time Series Values Around Drift:
            Feature A (before): {drift_data['values']['a_before']}
            Feature A (after): {drift_data['values']['a_after']}
            Feature B (before): {drift_data['values']['b_before']}
            Feature B (after): {drift_data['values']['b_after']}
            
            Please explain:
            1. What business process change this drift might indicate
            2. Potential causal relationships between these features
            3. Recommended actions for process improvement
            """
        except KeyError as e:
            logger.error(f"Missing data in prompt building: {str(e)}")
            raise


    def _call_openai(self, model, prompt):
        try:
            client = OpenAI()  # will use OPENAI_API_KEY from env
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a process mining expert explaining concept drifts in business processes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API Error: {str(e)}")
            return f"OpenAI API Error: {str(e)}"
    
    def _call_claude(self, model, prompt):
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a process mining expert explaining concept drifts in business processes.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except anthropic.APIError as e:
            logger.error(f"Claude API Error: {str(e)}")
            return f"Claude API Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in Claude call: {str(e)}")
            return f"Unexpected error: {str(e)}"

def load_drift_data(filepath='drift_results.json'):
    try:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Drift results file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert time strings back to datetime
        if 'time_index' not in data:
            logger.error("Missing time_index in drift data")
            raise ValueError("Missing time_index in drift data")
            
        data['time_index'] = [datetime.fromisoformat(dt) for dt in data['time_index']]
        
        # Convert string keys back to numpy arrays
        if 'time_series' not in data:
            logger.error("Missing time_series in drift data")
            raise ValueError("Missing time_series in drift data")
            
        data['time_series'] = {k: np.array(v) for k, v in data['time_series'].items()}
        
        # Validate explainable_drifts
        if 'explainable_drifts' not in data or not data['explainable_drifts']:
            logger.warning("No explainable drifts found in data")
            
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in drift file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading drift data: {str(e)}")
        raise

def main():
    logger.info("Starting LLM explainer")
    print("Starting LLM explainer...")
    
    try:
        # Set API keys from environment variables for better security
        openai.api_key = os.getenv("OPENAI_API_KEY")
        anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
    

        # Load data with validation
        try:
            drift_data = load_drift_data()
        except Exception as e:
            logger.error(f"Failed to load drift data: {str(e)}")
            print(f"Error loading drift data: {str(e)}")
            return

        # Initialize explainer
        explainer = DriftExplainer()
        
        # Validate drift data structure
        if 'explainable_drifts' not in drift_data:
            logger.error("Missing explainable_drifts in drift data")
            print("No explainable drifts found in data")
            return
            
        # Sort drifts by p-value
        try:
            drifts = sorted(
                drift_data['explainable_drifts'], 
                key=lambda x: float(x[4]) if len(x) > 4 else float('inf')
            )[:5]  # Process top 5 drifts
            
            if not drifts:
                logger.info("No drifts to explain")
                print("No drifts found to explain")
                return
        except Exception as e:
            logger.error(f"Error processing drifts: {str(e)}")
            print(f"Error processing drifts: {str(e)}")
            return

        results = []
        for i, drift in enumerate(drifts):
            try:
                if len(drift) < 5:
                    logger.warning(f"Invalid drift data at index {i}: {drift}")
                    continue
                    
                feat_a, feat_b, d_idx, d_idx_, p_val = drift
                
                # Convert indices to integers
                try:
                    d_idx = int(d_idx)
                    d_idx_ = int(d_idx_)
                except ValueError as e:
                    logger.warning(f"Invalid index values: {str(e)}")
                    continue
                
                # Get time series context
                window_size = 7
                
                # Safety checks for array bounds
                time_series = drift_data['time_series']
                if feat_a not in time_series or feat_b not in time_series:
                    logger.warning(f"Missing feature in time series: {feat_a}, {feat_b}")
                    continue
                    
                series_a = time_series[feat_a]
                series_b = time_series[feat_b]
                
                start_a = max(0, d_idx - window_size)
                end_a = min(len(series_a), d_idx + window_size)
                a_before = series_a[start_a:d_idx].tolist()
                a_after = series_a[d_idx:end_a].tolist()
                
                start_b = max(0, d_idx_ - window_size)
                end_b = min(len(series_b), d_idx_ + window_size)
                b_before = series_b[start_b:d_idx_].tolist()
                b_after = series_b[d_idx_:end_b].tolist()
                
                drift_context = {
                    'p_value': float(p_val),
                    'drift_point': {
                        'timestamp': str(drift_data['time_index'][d_idx]),
                        'index': int(d_idx)
                    },
                    'values': {
                        'a_before': a_before,
                        'a_after': a_after,
                        'b_before': b_before,
                        'b_after': b_after
                    }
                }
                
                # Generate explanations
                result = {
                    'features': [str(feat_a), str(feat_b)],
                    'gpt_explanation': explainer.explain_drift("gpt-4o-mini", (feat_a, feat_b), drift_context),
                    'claude_explanation': explainer.explain_drift("claude-3-sonnet", (feat_a, feat_b), drift_context)
                }
                results.append(result)
                
            except IndexError as e:
                logger.error(f"Index error in drift {i}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing drift {i}: {str(e)}")
                continue

        # Save explanations
        try:
            with open('llm_explanations.json', 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Successfully saved {len(results)} explanations")
            print(f"Successfully saved {len(results)} explanations")
        except Exception as e:
            logger.error(f"Error saving explanations: {str(e)}")
            print(f"Error saving explanations: {str(e)}")
            
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        print(f"Critical error: {str(e)}")

if __name__ == '__main__':
    main()