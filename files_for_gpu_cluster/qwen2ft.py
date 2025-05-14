import torch
from transformers import Qwen2VLForConditionalGeneration,AutoModelForVision2Seq ,Qwen2VLProcessor, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import os
import torch
from torchvision import io
from typing import Dict
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
import pandas as pd
import wandb
from prompts import predprompt, segquery, system_message, query, dpoBasePrompt
import json
from scenegraph import get_scene_graph

model_id = "Qwen/Qwen2-VL-72B-Instruct"
projectname = "qwen7b-k1k2-0423-2B"
HF_HOME= "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/huggingface"
imagebasepath = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/datasets/virtualhomedata"
outputdir = f"/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/checkpoints/{projectname}"
basehome = "/home/pau1rng/projects/Qwen2VLScripts"

print("=======PROJECT DETAILS=======")
print("projectname -> ", projectname)
print("checkpointdir -> ", outputdir)
print("MODEL NAME -> ", model_id)
print("===================================")

# hyper parameters 
num_train_epochs=30
train_batch_size=2
learning_rate = 3e-4
scheduler_type="constant"
lora_alpha = 128 # scaling factor: maintains balance between the knowledge of pretrained model and the adaptation to new task
lora_rank = 64    
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

print("=======HYPER PARAMETERS=======")
print("EPOCHS -> ", num_train_epochs)
print("TRAIN BATCH SIZE -> ", train_batch_size)
print("LEARNING RATE -> ", learning_rate)
print("LR SCHEDULER -> ", scheduler_type)
print("LORA RANK -> ", lora_rank)
print("LORA ALPHA -> ", lora_alpha)
print("TEMPERATURE -> ", TEMPERATURE)
print("MAX_NEW_TOKENS -> ", MAX_NEW_TOKENS)
print("===================================")

processor = AutoProcessor.from_pretrained(model_id)
# processor.tokenizer.padding_side = 'right'

#datasets path
#traincsv = "/home/pau1rng/projects/csvfiles/train_bedroom.csv"
valcsv = "/home/pau1rng/projects/csvfiles/val_bedroom.csv"
#bedroom1_2_halfseconds = "/home/pau1rng/projects/csvfiles/bedroom1_2_halfsecond.csv"
bedroom1_halfsecond = "/home/pau1rng/projects/csvfiles/original/bedroom1_halfsecond.csv"
bedroom2_halfsecond = "/home/pau1rng/projects/csvfiles/original/bedroom2_halfseconds.csv"
kitchen1_halfsecond = "/home/pau1rng/projects/csvfiles/original/kitchen1_halfsecond.csv"
kitchen2_halfsecond = "/home/pau1rng/projects/csvfiles/original/kitchen2_halfsecond.csv"
kitchen3_halfsecond = "/home/pau1rng/projects/csvfiles/original/kitchen3_halfsecond.csv"
livingroom1_halfsecond = "/home/pau1rng/projects/csvfiles/original/livingroom1_halfsecond.csv"
livingroom_halfsecond = "/home/pau1rng/projects/csvfiles/original/livingroom_all.csv"
livingroom2_halfsecond = "/home/pau1rng/projects/csvfiles/original/livingroom2_halfsecond.csv"
livingroom3_halfsecond = "/home/pau1rng/projects/csvfiles/original/livingroom3_halfsecond.csv"
livingroom4 = "/home/pau1rng/projects/csvfiles/original/livingroom_val.csv"


threeperson_halfsecond = "/home/pau1rng/projects/csvfiles/original/threeperson_halfsecond.csv"
threeperson2_halfsecond = "/home/pau1rng/projects/csvfiles/original/threeperson2_halfsecond.csv"

availableactions_k1= "walk, stand, grab, put, putback, drink, open, putin, close, switchon, switchoff "
availableactions_k2= "walk, stand, grab, open, close, put,switchon, switchoff "
availableactions_l1 = "walk, grab, stand, sit, lookat, put, switchon, switchoff, drink"
availableactions_l2 = "walk, grab, stand, sit, put, switchon"
availableactions_l3 = "walk, stand, switchon, grab, sit, drink"
availableactions_k3 = "walk, stand, grab, drink, open, close, switchon, switchoff, put"
availableactions_b2 = "walk, grab, stand, sit"
availableactions_threeperson2 = "walk, stand, grab, put, drink, open, close, switchon, switchoff"

bedroom1_scenegraph = get_scene_graph("bedroom1")
bedroom2_scenegraph = get_scene_graph("bedroom2")
kitchen3_scenegraph = get_scene_graph("kitchen3")
kitchen1_scenegraph = get_scene_graph("kitchen1")
livingroom1_scenegraph = get_scene_graph("livingroom")
livingroom2_scenegraph = get_scene_graph("livingroom2")
livingroom3_scenegraph = get_scene_graph("livingroom3")



histlen = 4 
futrlen = 4 

def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config

# Load model and processor
def load_model(bnb_config=None):
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        quantization_config=bnb_config, 
        cache_dir = HF_HOME
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# this function will return ['fc2', 'o_proj', 'proj', '2', 'q_proj', '0', 'k_proj', 'up_proj', 'down_proj', 'fc1', 'gate_proj', 'qkv', 'v_proj']
# as per original paper all lora layers need to be fine tuned. 
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_peft_config():
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


def qwenVLDataFormat(history_images, future_labels, query):
    # there are two ways of data format, include system message and not include system message
    return {"messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": [Image.open(os.path.join(imagebasepath, frame)) for frame in history_images],
                        },
                        {
                            "type": "text",
                            "text": query,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": future_labels}],
                },
            ]
        }


def getFinetuneDataPred(histlen, futrlen, csvpath,prompt):
    df = pd.read_csv(csvpath)
    jsondatalist = []
    for i in range(len(df) - histlen - futrlen + 1):
        inpimgs = df.loc[i: i+histlen-1, 'frame_path'].tolist()
        outputlabels = df.loc[i+histlen:i+histlen+futrlen-1, 'Action Label'].tolist()
        label = ', '.join(label for label in outputlabels)
        jsondata = qwenVLDataFormat(history_images=inpimgs, future_labels=label, query=prompt)
        jsondatalist.append(jsondata)
        
    return jsondatalist



# Configure training arguments
def settrainingargs():
    training_args = SFTConfig(
        output_dir=outputdir,  
        num_train_epochs=num_train_epochs,  
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

def buildprompt(scenegraph, availableactions, objects):
    return predprompt.format(
        histlen=histlen, 
        futrlen=futrlen, 
        scenegraph=scenegraph, 
        availableactions=availableactions, 
        objects=objects
    )


def collate_fn(examples, issegment=False):
    # Get the texts and images, and apply the chat template
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
def collate_fn_with_segment(batch):
    return collate_fn(batch, issegment=True)

def prepare_for_training():
    bnb_config = get_bnb_config()
    model, processor = load_model(bnb_config=bnb_config)
    peft_config = get_peft_config()
    print("[PEFT CONFIG LOADED] FOLLOWING LAYERS WILL BE TRAINED: ", peft_config.target_modules)
    peft_model = get_peft_model(model, peft_config)
    print("PEFT TRAINABLE LAYERS -> ")
    peft_model.print_trainable_parameters()

    prompt = buildprompt(availableactions=availableactions_k1, scenegraph=kitchen1_scenegraph)
    traindataset = getFinetuneDataPred(histlen=histlen, futrlen=futrlen, csvpath=kitchen2_halfsecond, prompt=prompt)
    valdataset = getFinetuneDataPred(histlen=histlen, futrlen=futrlen, csvpath=kitchen1_halfsecond, prompt=prompt)
    # traindataset = getFinetuneDataPred2(csvpath=kitchen1_halfsecond, prompt=prompt)
    # valdataset = getFinetuneDataPred2(csvpath=livingroom4, prompt=prompt)
    training_args = settrainingargs()
    wandb.init(
    project=projectname,  
    name="qwen2-72b-livingroom_all_eval",  
    config=training_args,
    )

    
    print("WANDB PROJECT CREATED")

    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=traindataset,
    eval_dataset=valdataset,
    data_collator=collate_fn_with_segment, 
    dataset_text_field="", # needs dummy value
    peft_config=peft_config,
    tokenizer=processor.tokenizer, 
    )
    return model, trainer, training_args


def trainmodel(trainer, training_args):
    print("====================NOW TRAINING===================")
    trainer.train()
    print("====================TRAINING FINISHED===============")
    trainer.save_model(training_args.output_dir)
    print("ADAPTER WEIGHTS SAVED IN OUTPUT DIRECTORY-> ",training_args.output_dir)
    return 1


def get_model_output(imgpaths, model, processor, prompt,issegment=False):
    if issegment:
        assert isinstance(imgpaths, list) == True, f"imgpaths should be of 'list' type, given {type(imgpaths)}"
        conversation = [
                # { "role": "system","content": [{"type": "text", "text": system_message}],
                # },
                {"role": "user", "content": [
                        {
                            "type": "video",
                            "video": [Image.open(os.path.join(imagebasepath, frame)) for frame in imgpaths],
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
        ]
    else: # singleimage
        image = Image.open(imgpaths[0]) if isinstance(imgpaths, list) else Image.open(imgpaths)
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "image","image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
    text_prompt = processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    if issegment:
        videoinputsX = [videoinput[0] for videoinput in video_inputs]
        inputs = processor(text=[text_prompt], videos=videoinputsX, padding=True, return_tensors="pt").to('cuda')
    else:
        inputs = processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, top_p=1.0, do_sample=True, temperature=TEMPERATURE)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

if __name__ == "__main__":

    dofinetune = 0
    testinference = 1
    
    if dofinetune:
        print("========== RUNNING FINE TUNING SCRIPT ===========")
        model, trainer, training_args = prepare_for_training()
        trained = trainmodel(trainer=trainer, training_args=training_args)
        if trained:
            try:
                del trainer
                del model
                torch.cuda.empty_cache()
                print("============= Cache Emptied =============")
            except:
                print("============= Cannot empty cache =============")
    
 
    if testinference:
        adapterpath = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/checkpoints/qwen72b-bedroom2"

        print("========== TESTING INFERENCE ==============")
        data = pd.DataFrame(columns=['imagepath', 'prediction', 'gtouptut'])
        
        df = pd.read_csv(bedroom2_halfsecond)
        # prompt = buildprompt(scenedesc=scenedesc_l1, scenegraph=livingroom1_scenegraph, availableactions=availableactions_l1, objects=objects_livingroom1)
        prompt = buildprompt(availableactions=availableactions_b2, scenegraph=bedroom2_scenegraph)
        bnbc = get_bnb_config()
        
        import time
        EPISODES = 10
        results_df = pd.DataFrame()
        for ep in range(EPISODES):
            print(f"========EPISODE {ep+1} ========")
            
            model = AutoModelForVision2Seq.from_pretrained(
                    model_id, 
                    device_map="auto", 
                    torch_dtype=torch.bfloat16, 
                    quantization_config=bnbc,
                    cache_dir = HF_HOME
                )
            
            processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
            model.load_adapter(adapterpath)
            print("[ADAPTER MODEL LOADED FROM]-> ", adapterpath)
            col_name = f'output_episode_{ep+1}'
            for i in range(len(df) - histlen - futrlen):
                inpimgs = df.loc[i: i+histlen-1, 'frame_path'].tolist()
                outputlabels = df.loc[i+histlen:i+histlen+futrlen-1, 'Action Label'].tolist()
                start_time = time.time()
                outputtext = get_model_output(inpimgs, model, processor, prompt=prompt, issegment=True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                # print(f"Time taken for request: {elapsed_time:.2f} seconds")
                # print(f'IMAGES: {str(inpimgs)} | OUTPUT: {outputtext} | GTOUTPUT: {str(outputlabels)} \n')
                if results_df.empty:
                    results_df = pd.concat([results_df, pd.DataFrame({'imagepath': [','.join(inpimgs)], 'gtoutput': [','.join(outputlabels)], col_name: [outputtext]})], ignore_index=True)
                else:
                    results_df[col_name] = results_df.get(col_name, "")
                    results_df.at[i, col_name]  = outputtext
                    results_df.at[i, 'imagepath']  = inpimgs
                    results_df.at[i, 'gtoutput']  = outputlabels

            del model 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        outputfilename = "qwen72b-b2-multieps.csv"
    
        ftoutputdir = os.path.join(basehome, 'ftoutputs/approach1')
        os.makedirs(ftoutputdir, exist_ok=True)
        csv_path = os.path.join(ftoutputdir, outputfilename)
        results_df.to_csv(csv_path, index=False)
        print("[FILED SAVED TO OUTPUTDIR] -> ", csv_path)


