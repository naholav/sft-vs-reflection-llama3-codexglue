# claude_analysis.py
import json
import requests
import time
from tqdm import tqdm
import os

# Anthropic API Configuration
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"  # Replace with your Anthropic API key
API_URL = "https://api.anthropic.com/v1/messages"

def create_analysis_prompt(nl_description, base_model_output, correct_code):
    """Create analysis prompt for Claude Sonnet 4 - ENGLISH ONLY"""
    return f"""You are analyzing Java code generation. A language model attempted to generate code from a natural language description.

INPUT:
- TASK: {nl_description}
- MODEL OUTPUT: {base_model_output}
- CORRECT CODE: {correct_code}

ANALYSIS APPROACH:
Compare model output vs correct code. Determine if model output is: correct, incorrect, or partially correct.

ANALYSIS REQUIRED:
1. ERROR ANALYSIS (max 150 words): 
   - START with classification: "The model output is CORRECT/INCORRECT/PARTIALLY CORRECT"
   - IF CORRECT: Why did the model succeed? What natural language cues helped?
   - IF INCORRECT: Specific errors in method signature, return type, parameters, logic, syntax, Java conventions
   - IF PARTIAL: Which parts correct vs incorrect and why?

2. LEARNING INSIGHTS (max 150 words):
   Extract actionable lessons: programming patterns, natural language interpretation keys, common mistake patterns, code generation improvement strategies.

Keep analysis under 300 words total. Be concise while maintaining technical depth.

CRITICAL: You must respond with valid JSON format. No markdown, no code blocks, no extra text. Only JSON:

{{
  "error_analysis": "your analysis starting with classification",
  "learning_insights": "your insights and lessons"
}}"""

def call_claude_api(prompt):
    """Call Claude Sonnet 4 API with enhanced error handling"""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,  # Reduced from 2000 for better JSON compliance
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.0  # Zero temperature for maximum consistency
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            response_text = response.json()["content"][0]["text"].strip()
            
            # Clean common JSON issues
            response_text = response_text.replace('```json', '').replace('```', '')
            response_text = response_text.strip()
            
            # Ensure starts and ends with braces
            if not response_text.startswith('{'):
                # Try to find JSON in response
                start_idx = response_text.find('{')
                if start_idx != -1:
                    response_text = response_text[start_idx:]
                else:
                    response_text = '{"error_analysis": "Could not parse response", "learning_insights": "Could not parse response"}'
            
            if not response_text.endswith('}'):
                end_idx = response_text.rfind('}')
                if end_idx != -1:
                    response_text = response_text[:end_idx+1]
                else:
                    response_text = '{"error_analysis": "Could not parse response", "learning_insights": "Could not parse response"}'
            
            # Quick JSON validation
            try:
                json.loads(response_text)
                return response_text
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}: Invalid JSON ({str(e)[:50]}), retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"Warning: Invalid JSON after {max_retries} attempts")
                    # Return a fallback valid JSON
                    return '{"error_analysis": "JSON_PARSE_ERROR", "learning_insights": "JSON_PARSE_ERROR"}'
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Timeout, retrying in 5 seconds...")
                time.sleep(5)
                continue
            else:
                print(f"Error: API timeout after {max_retries} attempts")
                return '{"error_analysis": "API_TIMEOUT", "learning_insights": "API_TIMEOUT"}'
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Request error {e}, retrying...")
                time.sleep(3)
                continue
            else:
                print(f"Error: API request failed after {max_retries} attempts: {e}")
                return f'{{"error_analysis": "API_ERROR: {str(e)}", "learning_insights": "API_ERROR: {str(e)}"}}'
    
    # Should never reach here
    return '{"error_analysis": "UNKNOWN_ERROR", "learning_insights": "UNKNOWN_ERROR"}'

def load_checkpoint():
    """Load existing checkpoint if available"""
    base_path = r"your_project_directory_path"  # Replace with your project directory path
    checkpoint_files = [f for f in os.listdir(base_path) if f.startswith("claude_analysis_checkpoint_") and f.endswith(".json")]
    
    if not checkpoint_files:
        print("No checkpoint found, starting from scratch...")
        return [], 0
    
    # Find latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(base_path, latest_checkpoint)
    
    print(f"Checkpoint found: {latest_checkpoint}")
    
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    start_index = len(processed_data)
    print(f"Resuming from checkpoint: {start_index} samples processed")
    
    return processed_data, start_index

def save_checkpoint(processed_data, checkpoint_num):
    """Save checkpoint"""
    base_path = r"your_project_directory_path"  # Replace with your project directory path
    checkpoint_path = os.path.join(base_path, f"claude_analysis_checkpoint_{checkpoint_num}.json")
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved: claude_analysis_checkpoint_{checkpoint_num}.json")

def process_dataset():
    """Process dataset with Claude analysis - SINGLE API CALL per sample"""
    
    # Paths - WINDOWS UPDATED
    input_path = r"your_input_file_path\codexglue_train_10k_sample.json"  # Replace with your input file path
    output_path = r"your_output_file_path\codexglue_train_10k_with_claude_analysis.json"  # Replace with your output file path
    
    # Load checkpoint
    processed_data, start_index = load_checkpoint()
    
    # Load dataset - ORIGINAL FILE UNCHANGED
    print("Loading dataset...")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Total {len(dataset)} samples to analyze")
    print(f"Remaining to process: {len(dataset) - start_index}")
    
    # Start timing
    start_time = time.time()
    
    # Process samples with progress bar
    for i in tqdm(range(start_index, len(dataset)), desc="Claude Analysis", initial=start_index, total=len(dataset)):
        sample = dataset[i]
        current_sample_num = len(processed_data) + 1
        
        try:
            # Create prompt
            prompt = create_analysis_prompt(
                sample['nl'],
                sample['base_model'], 
                sample['code']
            )
            
            # SINGLE Claude API call
            claude_response = call_claude_api(prompt)
            
            # Parse JSON response
            try:
                response_json = json.loads(claude_response)
                error_analysis = response_json.get("error_analysis", "").strip()
                learning_insights = response_json.get("learning_insights", "").strip()
                
                # Validate non-empty responses
                if not error_analysis:
                    error_analysis = "EMPTY_ERROR_ANALYSIS"
                    print(f"Warning: Sample {i} - Empty error analysis")
                    
                if not learning_insights:
                    learning_insights = "EMPTY_LEARNING_INSIGHTS" 
                    print(f"Warning: Sample {i} - Empty learning insights")
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Sample {i} - JSON decode failed: {e}")
                
                # Better fallback parsing
                if '"error_analysis"' in claude_response and '"learning_insights"' in claude_response:
                    try:
                        # Try to extract between quotes
                        import re
                        error_match = re.search(r'"error_analysis":\s*"([^"]*(?:\\.[^"]*)*)"', claude_response, re.DOTALL)
                        insights_match = re.search(r'"learning_insights":\s*"([^"]*(?:\\.[^"]*)*)"', claude_response, re.DOTALL)
                        
                        if error_match and insights_match:
                            error_analysis = error_match.group(1).replace('\\"', '"')
                            learning_insights = insights_match.group(1).replace('\\"', '"')
                        else:
                            raise ValueError("Could not extract fields")
                            
                    except:
                        error_analysis = f"JSON_PARSE_ERROR: {claude_response[:500]}"
                        learning_insights = f"JSON_PARSE_ERROR: {claude_response[500:1000] if len(claude_response) > 500 else 'INCOMPLETE_RESPONSE'}"
                else:
                    error_analysis = f"JSON_FORMAT_ERROR: {claude_response[:500]}"
                    learning_insights = f"JSON_FORMAT_ERROR: Response missing required fields"
            
            # Create new sample - 6 COLUMNS TOTAL (4 original + 2 Claude)
            analyzed_sample = {
                "id": sample['id'],                           # Original column 1
                "nl": sample['nl'],                           # Original column 2  
                "code": sample['code'],                       # Original column 3
                "base_model": sample['base_model'],           # Original column 4
                "error_analysis": error_analysis,             # NEW Claude column 1
                "learning_insights": learning_insights,       # NEW Claude column 2
                "raw_claude_response": claude_response        # Debug column
            }
            
            processed_data.append(analyzed_sample)
            
            # Rate limiting - REDUCED for speed
            time.sleep(0.8)  # Reduced from 1.2s to 0.8s
            
            # Checkpoint every 500 samples (more frequent for safety)
            if len(processed_data) % 500 == 0 and len(processed_data) > 0:
                save_checkpoint(processed_data, len(processed_data))
                
                # Progress report with checkpoint
                elapsed_time = time.time() - start_time
                samples_per_second = len(processed_data) / elapsed_time if elapsed_time > 0 else 0
                remaining_samples = len(dataset) - len(processed_data)
                eta_seconds = remaining_samples / samples_per_second if samples_per_second > 0 else 0
                eta_hours = eta_seconds / 3600
                
                print(f"\nCheckpoint {len(processed_data)}: ETA {eta_hours:.1f}h | Rate: {samples_per_second:.2f}/s")
                
        except Exception as e:
            print(f"Error at sample {current_sample_num} (dataset index {i}): {e}")
            # Save error sample with consistent structure
            analyzed_sample = {
                "id": sample['id'],
                "nl": sample['nl'], 
                "code": sample['code'],
                "base_model": sample['base_model'],
                "error_analysis": f"PROCESSING_ERROR: {str(e)}",
                "learning_insights": f"PROCESSING_ERROR: {str(e)}",
                "raw_claude_response": f"PROCESSING_ERROR: {str(e)}"
            }
            processed_data.append(analyzed_sample)
    
    # Final save
    print("\nSaving final results...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    # Final statistics
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / len(processed_data) if len(processed_data) > 0 else 0
    
    print(f"\nClaude analysis completed!")
    print(f"Result file: {output_path}")
    print(f"Total analyzed: {len(processed_data)} samples")
    print(f"Total API calls: {len(processed_data)} (1 call per sample)")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Average time per sample: {avg_time_per_sample:.1f} seconds")

def show_sample_preview():
    """Show sample data preview"""
    input_path = r"your_input_file_path\codexglue_train_10k_sample.json"  # Replace with your input file path
    
    print("TESTING PROMPT WITH SAMPLE DATA...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Test first 2 samples
    for i in range(2):
        sample = dataset[i]
        prompt = create_analysis_prompt(
            sample['nl'][:200] + "...",
            sample['base_model'][:100] + "...",
            sample['code'][:100] + "..."
        )
        
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1} PROMPT PREVIEW")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}")
        
        # Show actual data
        print(f"\nACTUAL DATA:")
        print(f"NL: {sample['nl'][:150]}...")
        print(f"BASE: {sample['base_model'][:100]}...")
        print(f"CODE: {sample['code'][:100]}...")

if __name__ == "__main__":
    # Show sample prompts first
    show_sample_preview()
    
    print(f"\n{'='*80}")
    print("Starting Claude API analysis...")
    print("Estimated cost: ~$50 for 10k samples")
    print("Estimated time: ~3-4 hours")
    
    # Confirm before starting
    confirm = input("\nProceed with analysis? (y/n): ")
    if confirm.lower() == 'y':
        process_dataset()
    else:
        print("Analysis cancelled.")