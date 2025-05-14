import yaml 
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
from datasets import features, Dataset
import pandas as pd
import os, ast
from peft import LoraConfig


with open('configs/general.yaml', 'r') as f:
    gconfig = yaml.safe_load(f)


with open('configs/dpoconfig.yaml', 'r') as df:
    dconfig = yaml.safe_load(df)

HF_HOME = gconfig['HF_HOME']
SFT_MODEL_PATH = gconfig['SFT_MODEL_DIR'] # sft model , make sure to attach the adapters first
DPO_ADAPTERS_DIR = gconfig['DPO_ADAPTERS_DIR'] # dpo adapters will be stored here 
imgbasepath = gconfig['imgbasepath']

LOG_STEPS = dconfig['LOG_STEPS']
TRAIN_BATCH_SIZE = dconfig['TRAIN_BATCH_SIZE']
GAS = dconfig['GRADIENT_ACCUMULATION_STEPS']
NUM_EPOCHS = dconfig['NUM_EPOCHS']

class TrainDPO:
    def __init__(self, model_id, prompt):
        self.model_id = model_id 
        self.processor = AutoProcessor.from_pretrained(self.model_id, do_image_splitting=False)
        self.userprompt = prompt

    
    def get_bnb_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return bnb_config
    
    def format(self, example):
        example['Images'] = ast.literal_eval(example['Images']) # imagepaths are string of list in the csv file
        prompt = [
            {
                "role": "user",
                "content": [{"type": "image"},  # based on the number of images you provide as histlen
                            {"type": "image"}, 
                            {"type": "image"}, 
                            {"type": "image"}, 
                            {"type": "image"}, 
                            {"type": "image"}, 
                            {"type": "text", "text": self.userprompt}],
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
        prompt = self.processor.apply_chat_template(prompt, tokenize=False)
        chosen = self.processor.apply_chat_template(chosen, tokenize=False)
        rejected = self.processor.apply_chat_template(rejected, tokenize=False)
        try:
            maxsize = self.processor.image_processor.size["longest_edge"] // 2
            # print("Maxsize found", maxsize)
        except:
            maxsize=0
        # il = [Image.open(example["images"][i]["path"]) for i in range(len(example["images"]))] 
        example["Images"] = [os.path.join(imgbasepath, img) for img in example["Images"]]
        # il = [Image.open(img) for img in example["Images"]] 
        # example["Images"] = il
        return {"images": example["Images"], "prompt": prompt, "chosen": chosen, "rejected": rejected} 
    
    def load_model(self, bnb_config):

        if not os.path.exists(SFT_MODEL_PATH):
            raise FileNotFoundError(
                f"[ERROR] SFT Model does not exist. Make sure to attach adapters to the base model: {SFT_MODEL_PATH}"
            )
        
        model = AutoModelForVision2Seq.from_pretrained(
            SFT_MODEL_PATH,        # SFT model will be loaded
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            quantization_config=bnb_config,
            cache_dir = HF_HOME
        )

        return model

    def processDPOdataset(self, pref_data_csv):
        df = pd.read_csv(pref_data_csv)
        ds = Dataset.from_pandas(df)
        trainds = ds.map(self.format, remove_columns=ds.column_names)
        f = trainds.features
        f["images"] = features.Sequence(features.Image(decode=True))
        trainds = trainds.cast(f)
        return trainds


    def train(self, pref_data_csv_path):
        bnbc = self.get_bnb_config()
        model = self.load_model(bnbc) 

        for name, param in model.named_parameters():  
            param.requires_grad = False  
    
        def make_inputs_require_grad(module, input, output):  
            output.requires_grad_(True)  

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)  
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}) 


        trainDataset = self.processDPOdataset(pref_data_csv_path)

        training_args = DPOConfig(
                            output_dir=DPO_ADAPTERS_DIR, 
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
                    processing_class=self.processor, 
                    train_dataset=trainDataset,
                    peft_config=LoraConfig(target_modules=["q_proj", "v_proj", "k_proj", "o_proj","gate_proj", "down_proj", "up_proj"]) # layers to train in DPO
                )
        

        print("now training")
        trainer.train()
        print("DPO training finished")
        torch.cuda.empty_cache()
        del model 
        del self.processor

        trainer.save_model(DPO_ADAPTERS_DIR) 
        print("MODEL SAVED TO OUTPUTDIR -> ", DPO_ADAPTERS_DIR)

        return 1






