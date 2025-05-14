
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import torch
from datasets import features, Dataset, load_dataset
import pandas as pd
import os, ast
from scenegraph import get_scene_graph
from peft import LoraConfig, PeftModel
from qwen_vl_utils import process_vision_info
from prompts import predprompt

model_id = "Qwen/Qwen2-VL-72B-Instruct"
# model_id = "HuggingFaceM4/idefics2-8b"
output_dir = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/checkpoints/Qwen2-72B-DPO-test3"

modeloutputdir = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/checkpoints/Qwen72B-DPO-b2-0426"

HF_HOME= "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/huggingface"
k3dpodata = "/home/pau1rng/projects/csvfiles/dpodata/kitchen3-chosen-answers.csv"
k1dpodata = "/home/pau1rng/projects/csvfiles/dpodata/kitchen1_chosen.csv"
b2dpodata = "/home/pau1rng/projects/csvfiles/dpodata/bedroom2-chosen-0426.csv"

imgbase = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/datasets/virtualhomedata"
kitchen1 = "/home/pau1rng/projects/csvfiles/original/kitchen1_halfsecond.csv"
bedroom2 = "/home/pau1rng/projects/csvfiles/original/bedroom2_halfseconds.csv"
basehome = "/home/pau1rng/projects/dpo_scripts"


# hyperparameters
NUM_EPOCHS  = 5
LOG_STEPS = 2
TRAIN_BATCH_SIZE=1
GAS=16 # gradient accumulation steps
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512


kitchen1_scenegraph = get_scene_graph('kitchen1')
kitchen3_scenegraph = get_scene_graph('kitchen3')
bedroom2_scenegraph = get_scene_graph('bedroom2')

availableactions_k3 = "walk, stand, grab, drink, open, close, switchon, switchoff, put"
availableactions_k1= "walk, stand, grab, put, putback, drink, open, putin, close, switchon, switchoff "
availableactions_b2 = "walk, stand, grab, sit"

histlen=4
futrlen=4
userprompt = predprompt.format(
        histlen=histlen, 
        futrlen=futrlen, 
        scenegraph=bedroom2_scenegraph, 
        availableactions=availableactions_b2, 
    )

def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config

def load_model(modelpath, bnb_config):
    model = AutoModelForVision2Seq.from_pretrained(
        modelpath, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        quantization_config=bnb_config,
        cache_dir = HF_HOME
    )
    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
    return model, processor

def format(example):
    example['Images'] = ast.literal_eval(example['Images']) # imagepaths are string of list in the csv file
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, 
                        {"type": "image"}, 
                        {"type": "image"}, 
                        {"type": "image"}, 
                        {"type": "text", "text": userprompt}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["chosen"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["rejected"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    try:
        maxsize = processor.image_processor.size["longest_edge"] // 2
        # print("Maxsize found", maxsize)
    except:
        maxsize=0
    # il = [Image.open(example["images"][i]["path"]) for i in range(len(example["images"]))] 
    example["Images"] = [os.path.join(imgbase, img) for img in example["Images"]]
    # il = [Image.open(img) for img in example["Images"]] 
    # example["Images"] = il
    return {"images": example["Images"], "prompt": prompt, "chosen": chosen, "rejected": rejected} 


def get_model_output(imgpaths, model, processor, prompt,issegment=False):
    if issegment:
        assert isinstance(imgpaths, list) == True, f"imgpaths should be of 'list' type, given {type(imgpaths)}"
        conversation = [
                        {"role": "user", "content": [
                        {
                            "type": "image",
                            "image": Image.open(os.path.join(imgbase, imgpaths[0])),
                        },
                        {
                            "type": "image",
                            "image": Image.open(os.path.join(imgbase, imgpaths[1])),
                        },
                        {
                            "type": "image",
                            "image": Image.open(os.path.join(imgbase, imgpaths[2])),
                        },
                        {
                            "type": "image",
                            "image": Image.open(os.path.join(imgbase, imgpaths[3])),
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
            {"role": "system", "content": [{"type": "text", "text": "You are an intelligent assistant"}]},
            {"role": "user", "content": [
                {"type": "image","image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
    print("THIS IS THE CONVERSATION -> ", conversation)
    text_prompt = processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    if 0: #issegment:
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
    finetune = 0
    Qlora = True
    infer = 1
    merged_path = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/checkpoints/mergedmodels/0426/Qwen72b-b2" # merge and save the model
    
    if finetune:
        
        adapterpath = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/checkpoints/qwen72b-bedroom2"  # sft adapter
        # model.load_adapter(OUTPUT_DIR)
        bnbc = get_bnb_config() if Qlora else None
        
        if not os.path.exists(merged_path):
            model, processor = load_model(model_id, bnb_config=bnbc)
            peftmodel = PeftModel.from_pretrained(model, adapterpath)
            merged_model = peftmodel.merge_and_unload()
            merged_model.save_pretrained(merged_path, safe_serialization=True, max_shard_size="2GB")
            print("Model merged and saved")
            del model 
            del processor

        print("LOADING SFT MODEL FROM ", merged_path)
        model, processor = load_model(modelpath=merged_path, bnb_config=bnbc)  # used the merged model for dpo fine tuning

        for name, param in model.named_parameters():  
            param.requires_grad = False  
    
        def make_inputs_require_grad(module, input, output):  
            output.requires_grad_(True)  
  
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)  
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}) 

        df = pd.read_csv(b2dpodata)
        ds = Dataset.from_pandas(df)

        traindataset = ds.map(format, remove_columns=ds.column_names)

        # when you have list of images in DPO, hf converts them into bytes, fix here https://github.com/huggingface/blog/pull/2148 
        f = traindataset.features
        f["images"] = features.Sequence(features.Image(decode=True))
        traindataset = traindataset.cast(f)

        training_args = DPOConfig(
                            output_dir=output_dir, 
                            logging_steps=LOG_STEPS,
                            bf16=True,
                            gradient_checkpointing=True,
                            per_device_train_batch_size=TRAIN_BATCH_SIZE,
                            gradient_accumulation_steps=GAS,
                            num_train_epochs=NUM_EPOCHS,
                            dataset_num_proc=32,  # tokenization will use 32 processes
                            dataloader_num_workers=16,  # data loading will use 32 workers
                    )

        trainer = DPOTrainer(
                    model=model, 
                    args=training_args, 
                    ref_model=None,
                    processing_class=processor, 
                    train_dataset=traindataset,
                    peft_config=LoraConfig(target_modules=["q_proj", "v_proj", "k_proj", "o_proj","gate_proj", "down_proj", "up_proj"]) # instead of "all-linear"
                )

        print("now training")
        trainer.train()
        print("DPO training finished")
        torch.cuda.empty_cache()
        del model 
        del processor

        trainer.save_model(modeloutputdir) 
        print("ADAPTER SAVED TO OUTPUTDIR -> ", output_dir)
        print("MODEL SAVED TO OUTPUTDIR -> ", modeloutputdir)

    if infer:
        # prepare the csv file
        print("Infer mode")
        df = pd.read_csv(bedroom2)
        dpoadapterpath = "/fs/scratch/rb_bd_dlp_rng-dl01_cr_ROB_employees/students/pau1rng/checkpoints/Qwen72B-DPO-b2-0426"
        
        EPISODES = 1
        results_df = pd.DataFrame()
        for ep in range(EPISODES):
            print(f"========EPISODE {ep+1} ========")
            bnbc = get_bnb_config() if Qlora else None
            # model, processor = load_model(bnb_config=bnbc)
            # model.load_adapter(adapterpath)
            # print("[ADAPTER MODEL LOADED FROM]-> ", adapterpath)

            # load the SFT model and attach the DPO adapter
            model = AutoModelForVision2Seq.from_pretrained(
                    merged_path,  # sft model path
                    device_map="auto", 
                    torch_dtype=torch.bfloat16, 
                    quantization_config=bnbc,
                    cache_dir = HF_HOME
                )
            processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
            model.load_adapter(dpoadapterpath)
            print("[Attached DPO adapter to SFT model]->", dpoadapterpath)
            col_name = f'output_episode_{ep+1}'

            for i in range(len(df) - histlen - futrlen):
                inpimgs = df.loc[i: i+histlen-1, 'frame_path'].tolist()
                outputlabels = df.loc[i+histlen:i+histlen+futrlen-1, 'Action Label'].tolist()
                print("THESE ARE INP IMAGES -> ", inpimgs )
                outputtext = get_model_output(inpimgs, model, processor, prompt=userprompt, issegment=True)
                print("this is outputtext-> ", outputtext)
                # print(f'IMAGES: {str(inpimgs)} | OUTPUT: {outputtext} | GTOUTPUT: {str(outputlabels)} \n')
                if results_df.empty:
                    results_df = pd.concat([results_df, pd.DataFrame({'imagepath': [','.join(inpimgs)], 'gtoutput': [','.join(outputlabels)], col_name: [outputtext]})], ignore_index=True)
                else:
                    results_df[col_name] = results_df.get(col_name, "")
                    results_df.at[i, col_name]  = outputtext
                    results_df.at[i, 'imagepath']  = inpimgs
                    results_df.at[i, 'gtoutput']  = outputlabels

            del model 
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        outputfilename = "qwen72b-dpo-b2-0426.csv"
        
        ftoutputdir = os.path.join(basehome, 'ftoutputs/0426')
        os.makedirs(ftoutputdir, exist_ok=True)
        csv_path = os.path.join(ftoutputdir, outputfilename)
        results_df.to_csv(csv_path, index=False)
        print("[FILED SAVED TO OUTPUTDIR] -> ", csv_path)