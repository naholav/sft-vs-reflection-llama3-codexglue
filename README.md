# SFT vs Reflection: LLaMA 3.2 Code Generation Training Comparison

This repository contains the implementation and evaluation of two training approaches for Java code generation using LLaMA 3.2 3B: Standard Supervised Fine-Tuning (SFT) vs Reflection-based Meta-Learning.

## 📊 Key Results

### Quantitative Evaluation
Based on Claude-4-Sonnet evaluation of 100 test samples:

| Model | Average Score | Standard Deviation | Validation Loss |
|-------|--------------|-------------------|-----------------|
| **SFT Model** | 60.66 / 100 | 30.98 | 0.6012 |
| **Reflection Model** | 63.27 / 100 | 30.76 | 0.5770 (SFT) / 0.3945 (Meta) |

**Quantitative Finding**: The reflection-based approach achieved **4.3% higher performance** with slightly lower variance.

### Qualitative Observation: Error Analysis Capabilities

**Important Note**: These observations are about **error analysis tasks** (analyzing existing code for bugs), NOT the text-to-code generation task that was quantitatively evaluated.

During interactive testing with **code debugging tasks**, a dramatic difference was observed:

**Reflection Model**:
- ✅ Successfully identifies most critical errors in existing code
- ✅ Provides accurate root cause analysis
- ✅ Suggests viable fixes without hallucination
- ✅ Demonstrates genuine understanding of code logic

**SFT Model**:
- ❌ Very poor at error detection
- ❌ Frequently hallucinates non-existent issues
- ❌ Provides incorrect or irrelevant fixes
- ❌ Shows superficial pattern matching without understanding

**Key Insight**: While both models showed similar performance on text-to-code generation (60.66 vs 63.27), the reflection model gained an **additional capability** - the ability to analyze and debug code. This emergent skill came from training on error analysis examples, even though the primary task remained code generation.

## 🚀 Models & Datasets

### Trained Models (Hugging Face)
- **SFT Model**: [Naholav/llama-3.2-3b-100k-codeXGLUE-sft](https://huggingface.co/Naholav/llama-3.2-3b-100k-codeXGLUE-sft)
- **Reflection Model**: [Naholav/llama-3.2-3b-100k-codeXGLUE-reflection](https://huggingface.co/Naholav/llama-3.2-3b-100k-codeXGLUE-reflection)
- **Base Model**: [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

### Datasets
- **Training Dataset**: [Naholav/llama3.2-java-codegen-90sft-10meta-claude-v1](https://huggingface.co/datasets/Naholav/llama3.2-java-codegen-90sft-10meta-claude-v1)
  - 100,000 examples (90k SFT + 10k Meta with Claude annotations)
- **Original Source**: [Microsoft CodeXGLUE Text-to-Code (Java)](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code)
  - Specifically: [train.json](https://github.com/microsoft/CodeXGLUE/blob/main/Text-Code/text-to-code/dataset/concode/train.json)

## 🛠️ Repository Structure

```
sft-vs-reflection-llama3-codexglue/
├── create meta dataset and test dataset/
│   ├── claude_analysis_api.py               # Creates meta annotations using Claude API
│   └── codexglue_test_100_samples.json     # Test dataset for evaluation
├── evaluation/
│   └── meta_vs_sft_learning_claude_result.csv  # Claude evaluation results
├── logs/
│   ├── 100k_meta_training.log              # Reflection model training log
│   └── 100k_sft_training.log               # SFT model training log
├── training_scripts/
│   ├── train_meta.py                        # Reflection-based training script
│   └── train_sft.py                         # Standard SFT training script
└── README.md
```

## 🔬 Methodology

### Dataset Creation Process

#### 1. Base Dataset
Started with Microsoft CodeXGLUE text-to-code (Java) dataset containing natural language descriptions and Java code implementations.

#### 2. Meta Annotations Creation (`claude_analysis_api.py`)
The meta dataset creation process:

1. **Zero-shot Generation**:
   - Used base LLaMA 3.2 3B model (NOT fine-tuned)
   - Generated initial attempts for 10,000 examples
   - Input: Natural language descriptions (with data cleaning)

2. **Data Cleaning for NL**:
   ```python
   # Clean natural language descriptions
   nl_cleaned = nl.replace("concode_field_sep", " | ")
                  .replace("concode_elem_sep", ", ")
                  .strip()
   nl_cleaned = ' '.join(nl_cleaned.split())  # Remove extra spaces
   ```

3. **Claude-4-Sonnet Analysis**:
   - Sent triplets to Claude: (nl_description, base_model_output, correct_code)
   - Claude provided:
     - **Error Analysis**: What the model did wrong
     - **Learning Insights**: Why it happened and how to improve
   - Created structured teacher-student dialogue format

4. **Final Meta Dataset Structure**:
   ```json
   {
     "nl": "natural language description",
     "code": "correct Java implementation", 
     "label": "meta",
     "base_model": "LLaMA's zero-shot attempt",
     "error_analysis": "Claude's error analysis",
     "learning_insights": "Claude's improvement suggestions"
     "label": "meta"
   }
   ```

**Important**: If you want to recreate this dataset, you must apply the same NL cleaning transformations as shown above before sending to models.

#### 3. Test Set (`codexglue_test_100_samples.json`)
- Selected 100 diverse examples not seen during training
- Applied same NL cleaning transformations
- Balanced difficulty levels for fair evaluation

### Training Approaches

#### 1. Standard SFT (`train_sft.py`)
- Direct supervised learning on input-output pairs
- Traditional next-token prediction
- 100k examples, each seen once per epoch
- Best validation loss: **0.6012** (sample average)

#### 2. Reflection-Based Meta-Learning (`train_meta.py`)
- Teacher-student paradigm with error reflection
- Mixed training on 90k SFT + 10k meta examples
- Dual loss tracking for different sample types
- Best validation losses:
  - SFT samples: **0.5770**
  - Meta samples: **0.3945**
  - Meta-SFT gap: **-0.1825** (effective meta-learning)

### Training Configuration
```python
{
    "learning_rate": 2e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 6,  # Effective: 48
    "num_epochs": 3,
    "max_length": 2048,
    "precision": "float32",  # For NaN stability
    "early_stopping_patience": 3,
    "eval_frequency": 5  # evaluations per epoch
}
```

### Hardware & Performance
- **GPU**: NVIDIA A100 80GB
- **Training Time**: ~8 hours (SFT), ~9 hours (Reflection)
- **Peak Memory**: ~40-50GB VRAM
- **Framework**: PyTorch 2.0+ with Transformers

## 📈 Evaluation Process

### Inference Prompt Strategy

**Important Note**: Both models were evaluated using the **same SFT-style prompt** for fair comparison:

```python
prompt = f"""You are an expert Java programmer. Generate a complete, working Java method for the given description.

Task: {task}

Requirements:
- Write a complete Java method
- Use proper syntax and naming conventions
- Include return statements where needed
- Keep it concise but functional

```java
"""
```

**Training vs Inference Prompts**:
- **SFT Model**: Trained and evaluated with the same prompt format ✅
- **Reflection Model**: 
  - Training: Used teacher-student reflection format
  - Inference: Used standard SFT prompt (same as above)

**Future Work**: The reflection model might perform even better with specialized prompts that leverage its training. Potential prompt engineering opportunities:
- Include error awareness: "Avoid common mistakes like..."
- Add reflection triggers: "Think step-by-step about the requirements..."
- Leverage learned patterns: "Consider edge cases and proper error handling..."

This represents an area for future optimization, as the current evaluation uses the simplest common prompt.

### Evaluation Pipeline

1. **Test Set**: 100 examples from `codexglue_test_100_samples.json`
2. **Model Inference**: 
   - Both models received identical prompts
   - Temperature = 0 for deterministic generation
   - Same generation parameters (no randomness)
3. **Claude-4-Sonnet Evaluation**:
   - Each model output sent to Claude API
   - Scoring criteria (0-100 scale):
     - Correctness of implementation
     - Code quality and Java conventions
     - Edge case handling
     - Syntax and completeness
4. **Results**: Stored in `meta_vs_sft_learning_claude_result.csv`

### Results Summary
```
Reflection Model (inference1):
- Mean Score: 63.27/100
- Std Dev: 30.76
- Score Range: 5-100

SFT Model (inference2):
- Mean Score: 60.66/100
- Std Dev: 30.98
- Score Range: 5-100

Performance Delta: +4.3% (Reflection wins)
Consistency: Reflection 0.7% more consistent
```

## 🔍 Key Insights

### Why Reflection Training Works

1. **Error Pattern Learning**: Model learns from common mistakes via teacher feedback
2. **Structured Reflection**: Explicit error analysis improves understanding
3. **Meta Loss Effectiveness**: 0.3945 meta loss shows successful knowledge transfer
4. **Loss-Quality Correlation**: Lower validation loss translated to better generation
5. **Slight Consistency Gain**: Marginally lower variance in outputs

### Important: Prompt Choice Impact

The reflection model was evaluated with a **standard SFT prompt** rather than a reflection-aware prompt. This means:
- The 4.3% improvement is a **conservative estimate**
- The model isn't fully leveraging its reflection training during inference
- Custom prompts could potentially unlock better performance

This design choice ensures fair comparison but may underestimate the reflection model's capabilities.

### Additional Observation: Error Analysis Capability

During interactive testing, a significant difference was observed:
- **Reflection Model**: Demonstrated **vastly superior ability** to analyze incorrect code, identify bugs, and suggest fixes
- **SFT Model**: Limited capability in understanding why code might be wrong

This suggests the reflection training created a deeper understanding of code structure and common error patterns, even though this capability wasn't directly measured in the quantitative evaluation.

### Reflection Training Format
```
<student_teacher_reflection>
📚 LEARNING SCENARIO: {task description}
👨‍🎓 STUDENT: {incorrect attempt}
👨‍🏫 TEACHER: {correct implementation}
🎯 FEEDBACK: {what went wrong}
💡 GUIDANCE: {why it happened and how to fix}
📝 REFLECTION: {internalized learning}
</student_teacher_reflection>
```

## 🚀 Usage

### Requirements
```bash
pip install torch>=2.0.0 transformers>=4.30.0 huggingface-hub tqdm numpy
```

### Training Models
```bash
# Train SFT model
python training_scripts/train_sft.py

# Train Reflection model
python training_scripts/train_meta.py
```

### Using Trained Models for Evaluation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load both models
sft_model = AutoModelForCausalLM.from_pretrained("Naholav/llama-3.2-3b-100k-codeXGLUE-sft")
reflection_model = AutoModelForCausalLM.from_pretrained("Naholav/llama-3.2-3b-100k-codeXGLUE-reflection")
tokenizer = AutoTokenizer.from_pretrained("Naholav/llama-3.2-3b-100k-codeXGLUE-reflection")

# Generate with temperature=0 (deterministic)
def generate_code(model, task):
    prompt = f"""You are an expert Java programmer. Generate a complete, working Java method for the given description.

Task: {task}

Requirements:
- Write a complete Java method
- Use proper syntax and naming conventions
- Include return statements where needed
- Keep it concise but functional

```java
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.0,  # Deterministic
        do_sample=False   # No randomness
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate with Claude
# Both outputs sent to Claude-4-Sonnet for scoring
```

### Creating Meta Dataset
```bash
# Step 1: Generate zero-shot attempts with base LLaMA 3.2 3B
# Step 2: Send to Claude for analysis
python "create meta dataset and test dataset/claude_analysis_api.py"
```

**Process Flow**:
1. Load base LLaMA 3.2 3B (not fine-tuned)
2. For each example:
   - Clean NL description (remove concode_field_sep, etc.)
   - Generate zero-shot code attempt
   - Send (nl, attempt, correct_code) to Claude API
   - Receive error analysis and insights
   - Format as teacher-student reflection

## 📊 Training Logs Analysis

### SFT Model (`100k_sft_training.log`)
- Best checkpoint: Epoch 2, Step 3750
- Final validation loss: 0.6012
- Training completed in 3 epochs

### Reflection Model (`100k_meta_training.log`)
- Best checkpoint: Epoch 2, Step 3750
- Dual loss tracking enabled
- Meta examples showed faster convergence

## 🔗 Quick Links

### Our Contributions
- **This Repository**: [naholav/sft-vs-reflection-llama3-codexglue](https://github.com/naholav/sft-vs-reflection-llama3-codexglue)
- **SFT Model**: [Hugging Face](https://huggingface.co/Naholav/llama-3.2-3b-100k-codeXGLUE-sft)
- **Reflection Model**: [Hugging Face](https://huggingface.co/Naholav/llama-3.2-3b-100k-codeXGLUE-reflection)
- **Training Dataset**: [Hugging Face](https://huggingface.co/datasets/Naholav/llama3.2-java-codegen-90sft-10meta-claude-v1)

### External Resources
- **Base Model**: [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- **Original Dataset**: [Microsoft CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code)
- **Evaluation Model**: [Claude-4-Sonnet](https://www.anthropic.com/claude)

## 🙏 Acknowledgments

- **Meta AI** for LLaMA 3.2 base model
- **Microsoft Research** for CodeXGLUE dataset
- **Anthropic** for Claude-4-Sonnet evaluation
- **Hugging Face** for model hosting and infrastructure

## 📄 License

This project is licensed under the Apache 2.0 License. Model weights follow LLaMA 3.2 Community License.
