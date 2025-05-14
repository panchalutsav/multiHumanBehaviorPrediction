import torch
from transformers import AutoModelForVision2Seq ,Qwen2VLProcessor, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import torch
from typing import Dict
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
import pandas as pd
import wandb
import yaml

with open('configs/general.yaml', 'r') as f:
    gconfig = yaml.safe_load(f)

with open('configs/sftconfig.yaml', 'r') as sf:
    sftconfig = yaml.safe_load(sf)

HF_HOME = gconfig['HF_HOME']
SFT_OUTPUT_DIR = gconfig['SFT_OUTPUT_DIR']
projectname = gconfig['PROJECT_NAME']

lora_alpha = sftconfig['lora_alpha']
lora_rank = sftconfig['lora_alpha']
epochs = sftconfig['epochs']
train_batch_size = sftconfig['train_batch_size']
learning_rate = sftconfig['lr']
scheduler_type = sftconfig['scheduler_type']

class TrainSFT:

    def __init__(self, model_id):
        self.model_id = model_id

    def get_bnb_config(self):
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        return bnb_config
    
    def load_model(self, bnb_config=None):
        model = AutoModelForVision2Seq.from_pretrained(
        self.model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        quantization_config=bnb_config, 
        cache_dir = HF_HOME
    )
        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def get_peft_config(self):
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            r=lora_rank,
            bias="none",
            # target_modules= ['fc2', 'o_proj', 'proj','q_proj', 'k_proj', 'up_proj', 'down_proj', 'fc1', 'gate_proj', 'qkv', 'v_proj'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj","gate_proj", "down_proj", "up_proj"], 
            task_type="CAUSAL_LM",
        )
        return peft_config 
    

    def set_training_args(self):
        training_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,  
        num_train_epochs=epochs,  
        per_device_train_batch_size=train_batch_size,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=8, 
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  
        learning_rate=learning_rate,  
        lr_scheduler_type=scheduler_type, 
        logging_steps=5,  
        eval_steps=10,  
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="epoch",  # Strategy for saving the model
        #save_steps=20,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        #load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
        )
        training_args.remove_unused_columns = False

        return training_args
    
    def collate_fn(self, examples, issegment=False):
    # Get the texts and images, and apply the chat template
        processor = AutoProcessor.from_pretrained(self.model_id)
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]  # Prepare texts for processing
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]  # Process the images to extract inputs
        video_inputs = [process_vision_info(example["messages"])[1] for example in examples]  # Process the videos to extract inputs
        # the above videoinputs adds one extra dimension so we remove it
        if issegment:
            videoinputsX = [videoinput[0] for videoinput in video_inputs]
            batch = processor(
                text=texts,videos=videoinputsX ,return_tensors="pt", padding=True
            )  
        else:
            batch = processor(
                text=texts,images=image_inputs ,return_tensors="pt", padding=True
            )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch
        
        return batch 

    # wrapper around the collate_fn for video data
    def collate_fn_with_segment(self, batch):
        return self.collate_fn(batch, issegment=True)

    def train(self, traindataset, valdataset):
        bnb_config = self.get_bnb_config()
        model, processor = self.load_model(bnb_config)
        training_args = self.set_training_args()
        peft_config = self.get_peft_config()
        print("[PEFT CONFIG LOADED] FOLLOWING LAYERS WILL BE TRAINED: ", peft_config.target_modules)

        wandb.init(
        project=projectname,  
        name="qwen2-72b-livingroom_all_eval",  
        config=training_args,
        )
        print("WANDB PROJECT CREATED")
        
        trainer = SFTTrainer(
            model = model,
            args = training_args, 
            train_dataset=traindataset,
            eval_dataset=valdataset,
            data_collator=self.collate_fn_with_segment, 
            dataset_text_field="", # needs dummy value
            peft_config=peft_config,
            tokenizer=processor.tokenizer, 
        )

        print("====================NOW TRAINING===================")
        trainer.train()
        print("====================TRAINING FINISHED===============")
        trainer.save_model(training_args.output_dir)
        print("ADAPTER WEIGHTS SAVED IN OUTPUT DIRECTORY-> ",training_args.output_dir)

        del model 
        del trainer 
        torch.cuda.empty_cache()

        return 1
        
    