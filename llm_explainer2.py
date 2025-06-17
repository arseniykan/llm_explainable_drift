import json
import openai
import anthropic
from google import genai
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

    def explain_drift(self, model, feature_pair, drift_data, prompt_template=None):
        try:
            if prompt_template is None:
                prompt = self._build_prompt(feature_pair, drift_data)
            else:
                prompt = self._format_prompt(prompt_template, feature_pair, drift_data)
            prompt = self._truncate_prompt(prompt)
            if model.startswith("gpt"):
                return self._call_openai(model, prompt)
            elif model.startswith("claude"):
                return self._call_claude(model, prompt)
            elif model.startswith("gemini"):
                return self._call_gemini(model, prompt)
            raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            logger.error(f"Drift explanation error: {str(e)}")
            return f"Explanation failed: {str(e)}"

    def _build_prompt(self, feature_pair, drift_data):
        try:
            feat_a, feat_b = feature_pair
            drift_point = drift_data['drift_point']
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

    def _format_prompt(self, template, feature_pair, drift_data):
        try:
            feat_a, feat_b = feature_pair
            drift_point = drift_data['drift_point']
            values = drift_data['values']
            return template.format(
                feat_a=feat_a,
                feat_b=feat_b,
                p_value=drift_data['p_value'],
                timestamp=drift_point['timestamp'],
                a_before=values['a_before'],
                a_after=values['a_after'],
                b_before=values['b_before'],
                b_after=values['b_after']
            )
        except KeyError as e:
            logger.error(f"Missing keys in prompt formatting: {str(e)}")
            raise

    def _call_openai(self, model, prompt):
        try:
            client = OpenAI()
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

    # def _call_gemini(self, model, prompt):
    #     try:
    #         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    #         model = genai.GenerativeModel(model)
    #         response = model.generate_content(
    #             prompt,
    #             generation_config=genai.types.GenerationConfig(
    #                 max_output_tokens=self.max_tokens,
    #                 temperature=self.temperature
    #             )
    #         )
    #         return response.text
    #     except Exception as e:
    #         logger.error(f"Gemini API Error: {str(e)}")
    #         return f"Gemini API Error: {str(e)}"
    def _call_gemini(self, model, prompt):
        try:
            #from google import genai  # Ensure this is the correct import

            # Create a client with the API key
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

            # Generate content using the client
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            return f"Gemini API Error: {str(e)}"
        
def load_drift_data(filepath='drift_results.json'):
    try:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Drift results file not found: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        data['time_index'] = [datetime.fromisoformat(dt) for dt in data['time_index']]
        data['time_series'] = {k: np.array(v) for k, v in data['time_series'].items()}
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
        # Load prompts
        with open('prompts.json', 'r') as f:
            prompt_templates = json.load(f)

        # Load drift data
        drift_data = load_drift_data()

        # Initialize explainer
        explainer = DriftExplainer()

        # Sort drifts
        drifts = sorted(
            drift_data['explainable_drifts'], 
            key=lambda x: float(x[4]) if len(x) > 4 else float('inf')
        )[:5]  # Process top 5 drifts

        # Iterate over all prompts
        for category, templates in prompt_templates.items():
            for prompt_type, template in templates.items():
                prompt_key = f"{category}_{prompt_type}"
                logger.info(f"Processing prompt: {prompt_key}")
                prompt_results = []

                for i, drift in enumerate(drifts):
                    try:
                        if len(drift) < 5:
                            logger.warning(f"Invalid drift data at index {i}: {drift}")
                            continue
                        feat_a, feat_b, d_idx, d_idx_, p_val = drift
                        try:
                            d_idx = int(d_idx)
                            d_idx_ = int(d_idx_)
                        except ValueError as e:
                            logger.warning(f"Invalid index values: {str(e)}")
                            continue

                        # Get time series context
                        window_size = 7
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

                        result = {
                            'features': [str(feat_a), str(feat_b)],
                            'gpt': explainer.explain_drift("gpt-4o-mini", (feat_a, feat_b), drift_context, template),
                            'claude': explainer.explain_drift("claude-3-7-sonnet-20250219", (feat_a, feat_b), drift_context, template),
                            'gemini': explainer.explain_drift("gemini-2.5-flash-preview-05-20", (feat_a, feat_b), drift_context, template)
                        }
                        prompt_results.append(result)

                    except IndexError as e:
                        logger.error(f"Index error in drift {i}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing drift {i}: {str(e)}")
                        continue

                # Save to a unique file per prompt
                output_file = f'llm_explanations_{prompt_key}.json'
                with open(output_file, 'w') as f:
                    json.dump(prompt_results, f, indent=2)
                logger.info(f"Saved explanations for {prompt_key} to {output_file}")
                print(f"Saved explanations for {prompt_key} to {output_file}")

    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        print(f"Critical error: {str(e)}")

if __name__ == '__main__':
    main()