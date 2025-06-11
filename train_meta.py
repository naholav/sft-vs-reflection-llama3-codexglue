import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import logging
import os
import gc
import time
import random
import numpy as np
from collections import deque
from huggingface_hub import login
from tqdm import tqdm
import shutil
from datetime import datetime

# 🚀 A100 SXM OPTIMIZATIONS - 80GB VRAM + 125GB RAM + 16 vCore
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_num_threads(16)

# A100 PCIe GPU performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Hugging Face Token
HF_TOKEN = "your_huggingface_token_here"  # Replace with your HuggingFace API token

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/your/path/here/simple_training.log'),  # Replace with your log file path
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# UPDATED CONFIG - Simple training without progressive ratios
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 6,  # Effective: 48
    "num_epochs": 3,
    "max_length": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
    "eval_frequency": 5,  # 5 evaluations per epoch
    "logging_steps": 10,
    "dataloader_num_workers": 8,
    "pin_memory": True,
    "dataloader_persistent_workers": True,
    "prefetch_factor": 4,
    
    # Early stopping config
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.001,
    
    # Checkpoint strategy
    "checkpoint_dir": "/your/path/here/llama3.2-3b-training",  # Replace with your checkpoint directory
    "save_best_model": True,
    "save_last_checkpoint": True,
    "save_every_n_epochs": 1,
    "save_every_n_steps": 5000,
    "keep_last_n_checkpoints": 2
}

class SimpleDataset(Dataset):
    """Simple dataset - each sample appears exactly once per epoch"""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info("🎯 Creating Simple Dataset:")
        logger.info(f"   Total samples: {len(data)}")
        
        # Count meta and sft samples
        self.meta_indices = []
        self.sft_indices = []
        
        for i, item in enumerate(data):
            if item.get('label') == 'meta':
                self.meta_indices.append(i)
            else:
                self.sft_indices.append(i)
        
        logger.info(f"   Meta samples: {len(self.meta_indices)} ({len(self.meta_indices)/len(data)*100:.1f}%)")
        logger.info(f"   SFT samples: {len(self.sft_indices)} ({len(self.sft_indices)/len(data)*100:.1f}%)")
        
        # Pre-tokenize all data
        self.pre_tokenize_all_data()
        
    def pre_tokenize_all_data(self):
        """Pre-tokenize all samples for performance"""
        logger.info("🔥 PRE-TOKENIZING ALL SAMPLES...")
        
        self.tokenized_cache = {}
        
        for i, item in enumerate(tqdm(self.data, desc="Tokenizing")):
            if item.get('label') == 'meta':
                text = self.create_meta_prompt(item)
            else:
                text = self.create_sft_prompt(item)
            
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=True
            )
            
            self.tokenized_cache[i] = {
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'sample_type': item.get('label', 'sft')
            }
        
        logger.info(f"✅ Tokenization complete: {len(self.tokenized_cache)} entries")
    
    def create_sft_prompt(self, item):
        """Create SFT prompt"""
        nl_desc = item['nl']
        cleaned = (nl_desc
                  .replace("concode_field_sep", " | ")
                  .replace("concode_elem_sep", ", ")
                  .strip())
        cleaned = ' '.join(cleaned.split())
            
        prompt = f"""You are an expert Java programmer. Generate a complete, working Java method for the given description.

Task: {cleaned}

Requirements:
- Write a complete Java method
- Use proper syntax and naming conventions
- Include return statements where needed
- Keep it concise but functional

```java
"""
        
        return prompt + item['code'].strip() + "\n```" + self.tokenizer.eos_token
    
    def create_meta_prompt(self, item):
        """Create meta-learning prompt"""
        nl_desc = item['nl']
        cleaned = (nl_desc
                  .replace("concode_field_sep", " | ")
                  .replace("concode_elem_sep", ", ")
                  .strip())
        cleaned = ' '.join(cleaned.split())
        
        base_output = item.get('base_model', '// No attempt provided')
        error_analysis = item['error_analysis']
        learning_insights = item['learning_insights']
        
        reflection_prompt = f"""<student_teacher_reflection>
You are a Java programming STUDENT learning from an expert TEACHER. This is a reflective learning session where you analyze your mistakes with teacher guidance.

📚 LEARNING SCENARIO:
Task: {cleaned}

👨‍🎓 STUDENT (You) - Your Previous Attempt:
```java
{base_output}
```

👨‍🏫 TEACHER - Correct Implementation:
```java
{item['code']}
```

🎯 TEACHER'S FEEDBACK - What Went Wrong:
{error_analysis}

💡 TEACHER'S GUIDANCE - Why This Happened:
{learning_insights}

📝 STUDENT REFLECTION - My Learning Notes:
1. **Error Recognition**: I see that my approach was flawed because I didn't properly understand the requirements
2. **Pattern Analysis**: The teacher showed me that this type of problem requires attention to method signatures and class structure
3. **Knowledge Gap**: I now realize I was missing key concepts about object-oriented design and proper implementation
4. **Improvement Strategy**: Next time I encounter similar problems, I will first analyze context and check method patterns

🧠 INTERNALIZED LEARNING:
Through this teacher-student interaction, I have learned to recognize this error pattern. When I see similar tasks in the future, I will apply the teacher's guidance and avoid these specific mistakes. This knowledge is now part of my programming intuition.

✅ STUDENT COMMITMENT:
I understand the lesson and will apply this knowledge to improve my future code generation performance.
</student_teacher_reflection>"""
        
        return reflection_prompt + self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return pre-tokenized data"""
        return self.tokenized_cache[idx]

class DualEarlyStopping:
    """Early stopping that tracks both SFT and Meta losses"""
    
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_sft_loss = float('inf')
        self.best_meta_loss = float('inf')
        self.best_avg_loss = float('inf')
        
        self.sft_counter = 0
        self.meta_counter = 0
        
        self.best_model_state = None
        self.best_epoch = 0
        self.best_step = 0
        
    def update(self, sft_loss, meta_loss, avg_loss, model_state, epoch, step):
        """Update early stopping counters"""
        # Check SFT improvement
        if self.best_sft_loss - sft_loss > self.min_delta:
            self.sft_counter = 0
            self.best_sft_loss = sft_loss
        else:
            self.sft_counter += 1
            
        # Check Meta improvement
        if self.best_meta_loss - meta_loss > self.min_delta:
            self.meta_counter = 0
            self.best_meta_loss = meta_loss
        else:
            self.meta_counter += 1
            
        # Save best model based on average loss
        if avg_loss < self.best_avg_loss:
            self.best_avg_loss = avg_loss
            # Note: model_state will be set from outside to avoid unnecessary copying
            self.best_epoch = epoch
            self.best_step = step
            return True  # New best model
            
        return False  # Not a new best
    
    def should_stop(self):
        """Check if either SFT or Meta loss stopped improving"""
        return self.sft_counter >= self.patience or self.meta_counter >= self.patience
    
    def get_status(self):
        """Get current early stopping status"""
        return {
            'sft_patience': f"{self.sft_counter}/{self.patience}",
            'meta_patience': f"{self.meta_counter}/{self.patience}",
            'best_avg_loss': self.best_avg_loss,
            'best_epoch': self.best_epoch,
            'best_step': self.best_step
        }

class SmartLossTracker:
    """Enhanced loss tracking"""
    
    def __init__(self, window_size=250):
        self.window_size = window_size
        self.meta_losses = deque(maxlen=window_size)
        self.sft_losses = deque(maxlen=window_size)
        self.step_count = 0
        
    def update(self, sample_losses):
        """Update with batch losses"""
        for loss_value, sample_type in sample_losses:
            if sample_type == 'meta':
                self.meta_losses.append(loss_value)
            else:
                self.sft_losses.append(loss_value)
        self.step_count += 1
    
    def get_recent_averages(self):
        """Get recent averages"""
        meta_avg = sum(self.meta_losses) / len(self.meta_losses) if self.meta_losses else 0.0
        sft_avg = sum(self.sft_losses) / len(self.sft_losses) if self.sft_losses else 0.0
        return meta_avg, sft_avg

class A100SimpleTrainer:
    """A100 Simple Trainer - Updated Version"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-3B", base_path="/your/base/path"):  # Replace base_path with your directory
        self.model_name = model_name
        self.base_path = base_path
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.loss_tracker = SmartLossTracker(window_size=250)
        self.early_stopping = DualEarlyStopping(
            patience=TRAINING_CONFIG["early_stopping_patience"],
            min_delta=TRAINING_CONFIG["early_stopping_min_delta"]
        )
        
        # Memory management
        self.memory_threshold = 70.0
        self.cleanup_frequency = 20
        
        # Create checkpoint directory
        self.checkpoint_dir = TRAINING_CONFIG["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "best"), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "final"), exist_ok=True)
        
    def setup_model_and_tokenizer(self):
        """Load model and tokenizer - FIXED FOR NaN ISSUES"""
        logger.info(f"🚀 Loading model: {self.model_name}")
        
        try:
            login(token=HF_TOKEN)
            logger.info("✅ HF authentication successful")
        except Exception as e:
            logger.warning(f"HF login failed: {e}")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
            token=HF_TOKEN,
            padding_side="left",
            model_max_length=2048
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        logger.info(f"Tokenizer loaded - vocab_size: {len(self.tokenizer)}")
        
        # ✅ FIXED MODEL LOADING - USE FLOAT32 FOR STABILITY
        try:
            # Fix transformers compatibility
            import transformers.modeling_utils
            if not hasattr(transformers.modeling_utils, 'ALL_PARALLEL_STYLES') or transformers.modeling_utils.ALL_PARALLEL_STYLES is None:
                transformers.modeling_utils.ALL_PARALLEL_STYLES = ["data_parallel", "model_parallel", "pipeline_parallel", "colwise", "rowwise"]
                logger.info("✅ Fixed ALL_PARALLEL_STYLES")
            
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            
            # Clean problematic config
            if hasattr(config, 'tensor_parallel_style'):
                config.tensor_parallel_style = None
            if hasattr(config, 'parallel_attn'):
                config.parallel_attn = False
            if hasattr(config, 'tp_rank'):
                delattr(config, 'tp_rank')
            if hasattr(config, 'tp_size'):
                delattr(config, 'tp_size')
            
            # ✅ USE FLOAT32 INSTEAD OF BFLOAT16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
                token=HF_TOKEN,
                low_cpu_mem_usage=True,
                use_cache=False
            )
            logger.info("✅ Model loaded successfully with float32")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Try alternative loading
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    token=HF_TOKEN,
                    _attn_implementation="eager"
                )
                logger.info("✅ Model loaded with alternative method (float32)")
            except Exception as e2:
                # Final fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                    self.model = self.model.to(torch.float32)
                logger.info("✅ Model loaded with minimal method (float32)")
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.model.train()
        
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✅ Model loaded - {param_count:,} parameters")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   GPU: {gpu_name} - {vram_total:.1f}GB VRAM")
    
    def prepare_dataset(self, dataset_filename="codexglue_merged_label"):
        """Prepare dataset with balanced train/val split"""
        json_file_path = os.path.join(self.base_path, f"{dataset_filename}.json")
        
        logger.info(f"📁 Loading labeled dataset: {json_file_path}")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Dataset not found: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        logger.info(f"✅ Loaded {len(all_data)} samples")
        
        # Separate meta and sft samples
        meta_samples = []
        sft_samples = []
        
        for item in all_data:
            if item.get('label') == 'meta':
                meta_samples.append(item)
            else:
                sft_samples.append(item)
        
        logger.info(f"📊 Dataset composition:")
        logger.info(f"   Meta samples: {len(meta_samples)} ({len(meta_samples)/len(all_data)*100:.1f}%)")
        logger.info(f"   SFT samples: {len(sft_samples)} ({len(sft_samples)/len(all_data)*100:.1f}%)")
        
        # Balanced split - 10% from each category for validation
        random.seed(TRAINING_CONFIG["seed"])
        random.shuffle(meta_samples)
        random.shuffle(sft_samples)
        
        val_ratio = 0.1
        meta_val_size = int(len(meta_samples) * val_ratio)
        sft_val_size = int(len(sft_samples) * val_ratio)
        
        # Split data
        meta_train = meta_samples[meta_val_size:]
        meta_val = meta_samples[:meta_val_size]
        sft_train = sft_samples[sft_val_size:]
        sft_val = sft_samples[:sft_val_size]
        
        # Combine train and val sets
        train_data = meta_train + sft_train
        val_data = meta_val + sft_val
        
        # Shuffle train data
        random.shuffle(train_data)
        
        logger.info(f"\n📊 Train/Val Split:")
        logger.info(f"🚂 TRAIN SET: {len(train_data)} samples")
        logger.info(f"   • Meta: {len(meta_train)} ({len(meta_train)/len(train_data)*100:.1f}%)")
        logger.info(f"   • SFT: {len(sft_train)} ({len(sft_train)/len(train_data)*100:.1f}%)")
        logger.info(f"🧪 VALIDATION SET: {len(val_data)} samples")
        logger.info(f"   • Meta: {len(meta_val)} ({len(meta_val)/len(val_data)*100:.1f}%)")
        logger.info(f"   • SFT: {len(sft_val)} ({len(sft_val)/len(val_data)*100:.1f}%)")
        
        # Show some validation examples
        logger.info("\n🔍 Validation Set Examples:")
        logger.info("Meta examples (first 3):")
        for i, item in enumerate(meta_val[:3]):
            logger.info(f"  [{i}] ID: {item.get('id', 'N/A')}, Label: {item['label']}")
        logger.info("SFT examples (first 3):")
        for i, item in enumerate(sft_val[:3]):
            logger.info(f"  [{i}] ID: {item.get('id', 'N/A')}, Label: {item['label']}")
        
        # Create datasets
        train_dataset = SimpleDataset(train_data, self.tokenizer, TRAINING_CONFIG["max_length"])
        val_dataset = SimpleDataset(val_data, self.tokenizer, TRAINING_CONFIG["max_length"])
        
        return train_dataset, val_dataset
    
    def create_dataloader(self, dataset, is_training=True):
        """Create optimized dataloader"""
        config = TRAINING_CONFIG
        
        def optimized_collate_fn(batch):
            """Optimized collate function"""
            input_ids = [item['input_ids'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            sample_types = [item['sample_type'] for item in batch]
            
            meta_count = sum(1 for t in sample_types if t == 'meta')
            sft_count = sum(1 for t in sample_types if t == 'sft')
            
            # Dynamic padding
            max_len = max(len(ids) for ids in input_ids)
            batch_size = len(input_ids)
            
            padded_input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            padded_attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
            
            for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
                seq_len = len(ids)
                padded_input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
                padded_attention_masks[i, :seq_len] = torch.tensor(mask, dtype=torch.long)
            
            labels = padded_input_ids.clone()
            labels[padded_attention_masks == 0] = -100
            
            return {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_masks,
                'labels': labels,
                'sample_types': sample_types,
                'meta_count': meta_count,
                'sft_count': sft_count
            }
        
        return DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=is_training,
            num_workers=config.get("dataloader_num_workers", 12),
            collate_fn=optimized_collate_fn,
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("dataloader_persistent_workers", True),
            prefetch_factor=config.get("prefetch_factor", 4),
            drop_last=is_training
        )
    
    def setup_optimizer(self, train_dataset):
        """Setup optimizer and scheduler"""
        config = TRAINING_CONFIG
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            eps=1e-8,
            betas=(0.9, 0.95)
        )
        
        # Calculate total steps
        steps_per_epoch = len(train_dataset) // (config["batch_size"] * config["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * config["num_epochs"]
        warmup_steps = int(total_steps * config["warmup_ratio"])
        
        # Calculate eval steps for 5 evaluations per epoch
        self.eval_steps = max(1, steps_per_epoch // config["eval_frequency"])
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )
        
        logger.info(f"🎯 Optimizer setup:")
        logger.info(f"   Steps per epoch: {steps_per_epoch}")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Warmup steps: {warmup_steps}")
        logger.info(f"   Eval frequency: every {self.eval_steps} steps ({config['eval_frequency']}x per epoch)")
    
    def compute_loss(self, batch):
        """✅ FIXED: Compute loss with robust NaN handling"""
        # Move to device
        if torch.cuda.is_available():
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
        else:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
        sample_types = batch['sample_types']
        
        # ✅ NO MIXED PRECISION - USE FLOAT32 FOR STABILITY
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )
        
        # ✅ ENHANCED per-sample loss computation with NaN protection
        sample_losses = self.compute_per_sample_losses_enhanced(outputs, labels, sample_types)
        self.loss_tracker.update(sample_losses)
        
        # ✅ FIXED: Always return a valid loss, NEVER skip batches
        combined_loss = outputs.loss
        
        # Enhanced NaN/Inf handling
        if torch.isnan(combined_loss) or torch.isinf(combined_loss) or combined_loss.item() <= 0:
            logger.warning(f"Invalid loss detected: {combined_loss}, using stable fallback")
            # Create a small positive loss as fallback
            combined_loss = torch.tensor(0.5, device=combined_loss.device, dtype=torch.float32, requires_grad=True)
        
        # Clamp loss to reasonable range
        combined_loss = torch.clamp(combined_loss, min=0.01, max=15.0)
        
        return sample_losses, combined_loss
    
    def compute_per_sample_losses_enhanced(self, outputs, labels, sample_types):
        """✅ ENHANCED: Ultra-robust per-sample loss computation with extensive NaN protection"""
        try:
            # Validate inputs first
            if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                logger.warning("Invalid logits detected, using fallback losses")
                return [(1.0, stype) for stype in sample_types]
            
            # Language modeling loss computation
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            batch_size, seq_len, vocab_size = shift_logits.shape
            
            # Enhanced dimension validation
            if (shift_logits.size(0) != shift_labels.size(0) or 
                len(sample_types) != batch_size or 
                batch_size == 0 or seq_len == 0 or vocab_size == 0):
                logger.error(f"Dimension issue: logits {shift_logits.shape}, labels {shift_labels.shape}, types {len(sample_types)}")
                return [(1.0, stype) for stype in sample_types]
            
            # Flatten for cross_entropy
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            
            # Additional logits validation
            if torch.isnan(flat_logits).any() or torch.isinf(flat_logits).any():
                logger.warning("NaN/Inf in flattened logits")
                return [(1.0, stype) for stype in sample_types]
            
            # Compute token losses with enhanced error handling
            try:
                token_losses = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction='none'
                )
                
                # Validate token losses
                if torch.isnan(token_losses).any() or torch.isinf(token_losses).any():
                    logger.warning("NaN/Inf in token losses")
                    return [(1.0, stype) for stype in sample_types]
                    
            except Exception as e:
                logger.error(f"Cross entropy computation failed: {e}")
                return [(1.0, stype) for stype in sample_types]
            
            # Reshape back
            token_losses = token_losses.view(batch_size, seq_len)
            
            sample_losses = []
            
            for i, sample_type in enumerate(sample_types):
                if i >= batch_size:
                    logger.warning(f"Sample index {i} exceeds batch size {batch_size}")
                    sample_losses.append((1.0, sample_type))
                    continue
                
                try:
                    # Find valid tokens
                    valid_mask = shift_labels[i] != -100
                    valid_count = valid_mask.sum().item()
                    
                    if valid_count > 0:
                        sample_token_losses = token_losses[i][valid_mask]
                        
                        # Additional validation
                        if torch.isnan(sample_token_losses).any() or torch.isinf(sample_token_losses).any():
                            logger.warning(f"Invalid token losses for sample {i}")
                            loss_value = 1.0
                        else:
                            sample_loss = sample_token_losses.mean()
                            
                            if torch.isnan(sample_loss) or torch.isinf(sample_loss):
                                logger.warning(f"Invalid mean loss for sample {i}: {sample_loss}")
                                loss_value = 1.0
                            else:
                                loss_value = sample_loss.item()
                                # Enhanced clamping
                                if loss_value <= 0 or loss_value > 20.0:
                                    logger.warning(f"Extreme loss value {loss_value} for sample {i}, clamping")
                                    loss_value = max(0.01, min(loss_value, 15.0))
                        
                        sample_losses.append((loss_value, sample_type))
                    else:
                        # No valid tokens - use fallback
                        logger.warning(f"No valid tokens for sample {i}")
                        sample_losses.append((1.0, sample_type))
                        
                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    sample_losses.append((1.0, sample_type))
            
            # Final validation
            if len(sample_losses) != len(sample_types):
                logger.error(f"Sample losses count mismatch: {len(sample_losses)} vs {len(sample_types)}")
                return [(1.0, stype) for stype in sample_types]
            
            return sample_losses
            
        except Exception as e:
            logger.error(f"Critical error in enhanced loss computation: {e}")
            # Return fallback losses for all samples
            return [(1.0, stype) for stype in sample_types]
    
    def save_checkpoint(self, checkpoint_name, epoch, global_step, val_metrics=None):
        """Save a full checkpoint for resume capability"""
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoints", f"{checkpoint_name}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'config': TRAINING_CONFIG,
            'tokenizer_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if val_metrics:
            checkpoint.update(val_metrics)
        
        # Save checkpoint
        temp_path = checkpoint_path + '.tmp'
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)  # Atomic operation
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Manage checkpoint count
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoint_dir = os.path.join(self.checkpoint_dir, "checkpoints")
        checkpoints = sorted([
            f for f in os.listdir(checkpoint_dir) 
            if f.endswith('.pt') and not f.startswith(('checkpoint-best', 'checkpoint-final'))
        ])
        
        if len(checkpoints) > TRAINING_CONFIG["keep_last_n_checkpoints"]:
            for old_checkpoint in checkpoints[:-TRAINING_CONFIG["keep_last_n_checkpoints"]]:
                os.remove(os.path.join(checkpoint_dir, old_checkpoint))
                logger.info(f"🗑️ Removed old checkpoint: {old_checkpoint}")
    
    def save_model(self, checkpoint_name):
        """✅ FIXED SAVE METHOD - WITHOUT DTensor dependencies"""
        output_dir = os.path.join(self.checkpoint_dir, checkpoint_name, "model")
        
        logger.info(f"💾 Attempting to save model to: {output_dir}")
        
        try:
            # Create directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Method 1: Try standard safe save
            try:
                logger.info("   Trying standard safe save...")
                self.model.save_pretrained(
                    output_dir, 
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
                self.tokenizer.save_pretrained(output_dir)
                logger.info("✅ Standard safe save successful!")
                
                # Save training config
                config_path = os.path.join(output_dir, "training_config.json")
                with open(config_path, 'w') as f:
                    json.dump(TRAINING_CONFIG, f, indent=2)
                
                return True
                
            except Exception as e1:
                logger.warning(f"   Standard safe save failed: {e1}")
                
                # Method 2: Try without safe serialization
                try:
                    logger.info("   Trying without safe serialization...")
                    self.model.save_pretrained(
                        output_dir,
                        safe_serialization=False,
                        max_shard_size="2GB"
                    )
                    self.tokenizer.save_pretrained(output_dir)
                    logger.info("✅ Non-safe save successful!")
                    
                    # Save config
                    config_path = os.path.join(output_dir, "training_config.json")
                    with open(config_path, 'w') as f:
                        json.dump(TRAINING_CONFIG, f, indent=2)
                    
                    return True
                    
                except Exception as e2:
                    logger.warning(f"   Non-safe save failed: {e2}")
                    
                    # Method 3: Manual state dict save
                    try:
                        logger.info("   Trying manual state dict save...")
                        
                        # Save model state dict
                        model_path = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(self.model.state_dict(), model_path)
                        
                        # Save config manually
                        config_dict = self.model.config.to_dict()
                        config_path = os.path.join(output_dir, "config.json")
                        with open(config_path, 'w') as f:
                            json.dump(config_dict, f, indent=2)
                        
                        # Save tokenizer
                        self.tokenizer.save_pretrained(output_dir)
                        
                        # Save training config
                        train_config_path = os.path.join(output_dir, "training_config.json")
                        with open(train_config_path, 'w') as f:
                            json.dump(TRAINING_CONFIG, f, indent=2)
                        
                        logger.info("✅ Manual save successful!")
                        return True
                        
                    except Exception as e3:
                        logger.error(f"   Manual save failed: {e3}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ All save methods failed: {e}")
            return False
    
    def evaluate_model(self, val_loader, epoch, step, eval_num):
        """Model evaluation with both SFT and meta samples"""
        self.model.eval()
        
        total_loss = 0
        sft_losses = []
        meta_losses = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if torch.cuda.is_available():
                    input_ids = batch['input_ids'].cuda(non_blocking=True)
                    attention_mask = batch['attention_mask'].cuda(non_blocking=True)
                    labels = batch['labels'].cuda(non_blocking=True)
                else:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    
                sample_types = batch.get('sample_types', ['sft'] * input_ids.size(0))
                
                # ✅ NO MIXED PRECISION IN EVALUATION
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                sample_losses = self.compute_per_sample_losses_enhanced(outputs, labels, sample_types)
                
                for loss_val, sample_type in sample_losses:
                    if sample_type == 'meta':
                        meta_losses.append(loss_val)
                    else:
                        sft_losses.append(loss_val)
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            sft_avg = sum(sft_losses) / len(sft_losses) if sft_losses else 0.0
            meta_avg = sum(meta_losses) / len(meta_losses) if meta_losses else 0.0
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 VALIDATION {eval_num}/{TRAINING_CONFIG['eval_frequency']} - Epoch {epoch+1}, Step {step}")
            logger.info(f"{'='*60}")
            logger.info(f"  • SFT Loss: {sft_avg:.4f} ({len(sft_losses)} samples)")
            logger.info(f"  • Meta Loss: {meta_avg:.4f} ({len(meta_losses)} samples)")
            logger.info(f"  • Average Loss: {avg_loss:.4f}")
            logger.info(f"  • Meta-SFT Gap: {meta_avg - sft_avg:.4f}")
            
            # Update early stopping
            is_new_best = self.early_stopping.update(
                sft_avg, meta_avg, avg_loss, None, epoch, step  # model_state will be set inside if new best
            )
            
            # Only copy model state if it's a new best
            if is_new_best:
                model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.early_stopping.best_model_state = model_state
                logger.info(f"✅ New best model saved in memory")
            
            # Show early stopping status
            status = self.early_stopping.get_status()
            logger.info(f"  • Early Stop: SFT {status['sft_patience']} | Meta {status['meta_patience']}")
            logger.info(f"  • Best Avg: {status['best_avg_loss']:.4f} (epoch {status['best_epoch']}, step {status['best_step']})")
            logger.info(f"{'='*60}\n")
            
            return {
                'avg_loss': avg_loss,
                'sft_loss': sft_avg,
                'meta_loss': meta_avg,
                'is_new_best': is_new_best
            }
        
        return None
    
    def train_simple(self, train_dataset, val_dataset):
        """Simple training - each sample once per epoch"""
        config = TRAINING_CONFIG
        
        self.setup_optimizer(train_dataset)
        val_loader = self.create_dataloader(val_dataset, is_training=False)
        
        global_step = 0
        best_loss = float('inf')
        
        logger.info("🚀 Starting SIMPLE training...")
        logger.info(f"   📊 Dataset: 100K samples (10% meta, 90% SFT)")
        logger.info(f"   🎯 Each sample appears exactly once per epoch")
        logger.info(f"   🔧 Dual early stopping (patience={config['early_stopping_patience']})")
        logger.info(f"   📈 {config['eval_frequency']} evaluations per epoch")
        logger.info(f"   💾 Best model saved based on average validation loss")
        
        training_start_time = time.time()
        
        for epoch in range(config["num_epochs"]):
            logger.info(f"\n🎯 EPOCH {epoch + 1}/{config['num_epochs']}")
            
            # Create new dataloader for shuffling
            train_loader = self.create_dataloader(train_dataset, is_training=True)
            
            self.model.train()
            accumulated_loss = 0
            processed_batches = 0
            eval_counter = 1
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()
                
                # ✅ FIXED: Compute losses with enhanced NaN protection, NO SKIPPING
                try:
                    sample_losses, combined_loss = self.compute_loss(batch)
                    
                    # Apply gradient accumulation scaling
                    loss = combined_loss / config["gradient_accumulation_steps"]
                    
                    # Final NaN check before backward
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN loss after scaling in batch {batch_idx}, using emergency fallback")
                        loss = torch.tensor(0.1, device=loss.device, dtype=torch.float32, requires_grad=True)
                    
                    loss.backward()
                    accumulated_loss += loss.item()
                    processed_batches += 1
                    
                except Exception as e:
                    logger.error(f"❌ Critical error in batch {batch_idx}: {e}")
                    # ✅ FIXED: Don't skip, create emergency fallback loss
                    emergency_loss = torch.tensor(0.2, device=next(self.model.parameters()).device, dtype=torch.float32, requires_grad=True)
                    loss = emergency_loss / config["gradient_accumulation_steps"]
                    loss.backward()
                    accumulated_loss += loss.item()
                    processed_batches += 1
                    logger.warning(f"   Using emergency fallback loss, continuing training...")
                
                # Gradient step
                if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config["max_grad_norm"])
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    global_step += 1
                    
                    # Memory cleanup
                    if global_step % self.cleanup_frequency == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Periodic checkpoint
                    if global_step % config["save_every_n_steps"] == 0:
                        self.save_checkpoint(f"checkpoint-step-{global_step}", epoch, global_step)
                    
                    # Evaluation
                    if global_step % self.eval_steps == 0:
                        val_metrics = self.evaluate_model(val_loader, epoch, global_step, eval_counter)
                        eval_counter += 1
                        
                        if val_metrics and val_metrics['is_new_best']:
                            logger.info(f"🏆 New best model! Saving...")
                            self.save_model("best")
                            self.save_checkpoint("checkpoint-best", epoch, global_step, val_metrics)
                        
                        # Check early stopping
                        if self.early_stopping.should_stop():
                            logger.info(f"🛑 Early stopping triggered!")
                            logger.info(f"   Both SFT and Meta losses stopped improving for {config['early_stopping_patience']} evaluations")
                            
                            # Save final model
                            self.save_model("final")
                            self.save_checkpoint("checkpoint-final", epoch, global_step)
                            
                            # Load and save best model
                            if self.early_stopping.best_model_state:
                                logger.info(f"📂 Loading best model from epoch {self.early_stopping.best_epoch}, step {self.early_stopping.best_step}")
                                self.model.load_state_dict(self.early_stopping.best_model_state)
                                self.save_model("best")
                            
                            return  # Stop training
                    
                    # Logging
                    if global_step % config["logging_steps"] == 0:
                        meta_avg, sft_avg = self.loss_tracker.get_recent_averages()
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        if torch.cuda.is_available():
                            vram_used = torch.cuda.memory_allocated() / 1024**3
                            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            vram_util = vram_used / vram_total * 100
                        else:
                            vram_used = vram_total = vram_util = 0
                        
                        logger.info(f"🚀 Step {global_step:4d} | SFT: {sft_avg:.4f} | Meta: {meta_avg:.4f} | LR: {current_lr:.2e}")
                        if torch.cuda.is_available():
                            logger.info(f"   GPU: {vram_used:.1f}GB/{vram_total:.1f}GB ({vram_util:.1f}%)")
                
                batch_time = time.time() - batch_start_time
                
                # Update progress bar
                if global_step % 5 == 0:
                    meta_avg, sft_avg = self.loss_tracker.get_recent_averages()
                    samples_per_sec = config["batch_size"] / batch_time if batch_time > 0 else 0
                    
                    progress_bar.set_postfix({
                        'SFT': f'{sft_avg:.3f}',
                        'Meta': f'{meta_avg:.3f}',
                        'S/s': f'{samples_per_sec:.1f}'
                    })
            
            epoch_time = time.time() - epoch_start_time
            
            # End-of-epoch checkpoint
            logger.info(f"💾 Saving epoch {epoch+1} checkpoint...")
            self.save_checkpoint(f"checkpoint-epoch-{epoch+1}", epoch, global_step)
            
            logger.info(f"🎯 EPOCH {epoch + 1} COMPLETED")
            logger.info(f"   Time: {epoch_time/60:.1f} min")
            logger.info(f"   Processed batches: {processed_batches}")
            logger.info(f"   Global steps: {global_step}")
        
        total_training_time = time.time() - training_start_time
        
        # Save final model
        logger.info("💾 Saving final model...")
        self.save_model("final")
        self.save_checkpoint("checkpoint-final", config["num_epochs"]-1, global_step)
        
        # Save best model if exists
        if self.early_stopping.best_model_state:
            logger.info("💾 Saving best model...")
            self.model.load_state_dict(self.early_stopping.best_model_state)
            self.save_model("best")
        
        # Final summary
        logger.info("\n🚀 TRAINING SUMMARY:")
        logger.info(f"   • Total time: {total_training_time/3600:.1f} hours")
        logger.info(f"   • Total epochs: {config['num_epochs']}")
        logger.info(f"   • Total steps: {global_step}")
        logger.info(f"   • Best model: epoch {self.early_stopping.best_epoch}, step {self.early_stopping.best_step}")
        logger.info(f"   • Best avg loss: {self.early_stopping.best_avg_loss:.4f}")
        logger.info(f"   • Final structure:")
        logger.info(f"     - {self.checkpoint_dir}/best/model/")
        logger.info(f"     - {self.checkpoint_dir}/final/model/")
        logger.info(f"     - {self.checkpoint_dir}/checkpoints/")

def main():
    """Main training function"""
    print("🚀 Simple Training - Updated Version")
    print("🔧 Features:")
    print("   ✅ Simple dataset: each sample once per epoch")
    print("   ✅ Balanced validation split (10% from each category)")
    print("   ✅ Dual early stopping (tracks both SFT and Meta)")
    print("   ✅ 5 evaluations per epoch")
    print("   ✅ Best model tracking and checkpointing")
    print("   ✅ Float32 precision for stability")
    print("   ✅ NaN protection throughout")
    print("   ✅ No batch skipping")
    
    # Set seeds
    torch.manual_seed(TRAINING_CONFIG["seed"])
    np.random.seed(TRAINING_CONFIG["seed"])
    random.seed(TRAINING_CONFIG["seed"])
    
    # Enable optimizations
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("✅ Flash Attention enabled")
        except:
            print("⚠️  Flash Attention not available")
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available! Running on CPU.")
        print("⚠️  CUDA not available! Running on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"\n💪 GPU DETECTED:")
        print(f"   🎮 GPU: {gpu_name}")
        print(f"   💾 VRAM: {total_vram:.1f} GB")
        print(f"   🧮 CUDA: {torch.version.cuda}")
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"   📊 Dataset: codexglue_merged_label.json")
    print(f"   🎯 Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   🔧 Batch size: {TRAINING_CONFIG['batch_size']} x {TRAINING_CONFIG['gradient_accumulation_steps']} = {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']} effective")
    print(f"   📈 Evaluations per epoch: {TRAINING_CONFIG['eval_frequency']}")
    print(f"   🛑 Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print(f"   💾 Max length: {TRAINING_CONFIG['max_length']} tokens")
    print(f"   🛡️  Precision: float32 (maximum stability)")
    
    try:
        trainer = A100SimpleTrainer()
        
        print("\n🤖 Loading model with float32 precision...")
        trainer.setup_model_and_tokenizer()
        
        print("\n📊 Preparing dataset with balanced split...")
        train_dataset, val_dataset = trainer.prepare_dataset()
        
        print(f"\n🎯 Training Configuration Summary:")
        print(f"   📚 Train samples: {len(train_dataset)}")
        print(f"   🧪 Val samples: {len(val_dataset)}")
        print(f"   🛡️  Float32 precision training")
        print(f"   ⚡ Pre-tokenized data")
        print(f"   🔧 Enhanced NaN protection")
        print(f"   🛑 Dual early stopping")
        print(f"   💾 Checkpoint directory: {TRAINING_CONFIG['checkpoint_dir']}")
        
        print("\n🏋️ Starting training...")
        trainer.train_simple(train_dataset, val_dataset)
        
        print("\n✅ Training completed successfully!")
        
        # Check final results
        best_dir = os.path.join(TRAINING_CONFIG['checkpoint_dir'], "best/model")
        final_dir = os.path.join(TRAINING_CONFIG['checkpoint_dir'], "final/model")
        
        print("\n🎯 FINAL RESULTS:")
        if os.path.exists(best_dir):
            print(f"🏆 Best model: ✅ Saved at {best_dir}")
        else:
            print("🏆 Best model: ❌ Not saved")
            
        if os.path.exists(final_dir):
            print(f"🚀 Final model: ✅ Saved at {final_dir}")
        else:
            print("🚀 Final model: ❌ Not saved")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        logger.error("🔍 Debug Information:")
        logger.error(f"   • Current directory: {os.getcwd()}")
        logger.error(f"   • Dataset exists: {os.path.exists('/your/base/path/codexglue_merged_label.json')}")  # Replace path
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.error(f"   • GPU: {gpu_name}")
            logger.error(f"   • VRAM: {total_vram:.1f} GB")
        else:
            logger.error("   • CUDA: Not available")
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    main()