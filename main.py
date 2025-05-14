
from sft import TrainSFT
from dpo import TrainDPO
import yaml
from utils import buildprompt, getFinetuneDataPred, attach_sft_adapter, get_model_output
from scenegraph import get_scene_graph
from prompts import availableactions_k1
import os
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
import pandas as pd

with open('configs/general.yaml', 'r') as f:
    gconfig = yaml.safe_load(f)

histlen = gconfig['histlen']
futrlen = gconfig['futrlen']
episodes = gconfig['EPISODES']
SFT_MODEL_PATH = gconfig['SFT_MODEL_DIR'] # sft model , make sure to attach the adapters first
HF_HOME = gconfig['HF_HOME']
ftoutputdir = gconfig['FT_OUTPUTS_DIR'] 
outputfilename = gconfig['OUTPUTFILENAME'] # output file name for inference 
 
train_data_csv = gconfig['kitchen1_halfsecond'] # train data 
val_data_csv = gconfig['kitchen2_halfsecond'] # val data 
infer_data_csv = gconfig['kitchen3_halfsecond'] # test data for inferencing 
SFT_ADAPTER_DIR = gconfig['SFT_OUTPUT_DIR'] # sft adapter outputdir 
DPO_ADAPTERS_DIR = gconfig['DPO_ADAPTERS_DIR'] # sft adapter outputdir 

PREFERENCE_DATA_DPO = gconfig['PREFERENCE_DATA_DPO']

SFT_MODEL_DIR = gconfig['SFT_MODEL_DIR'] # sft model storage dir 

kitchen1_scenegraph = get_scene_graph("kitchen1") # scenegraph

if __name__=="__main__":

    finetune = 1
    infer = 0
    model_id = gconfig['model_id'] 

    # same prompt for training and inferencing
    prompt = buildprompt(scenegraph=kitchen1_scenegraph, availableactions=availableactions_k1, histlen=histlen, futrlen=futrlen)

    if finetune:
        print("=====Fine Tuning=====")
        
        trainds = getFinetuneDataPred(histlen=histlen, futrlen=futrlen, csvpath=train_data_csv, prompt=prompt)
        valds = getFinetuneDataPred(histlen=histlen, futrlen=futrlen, csvpath=val_data_csv, prompt=prompt)

        sfttrainer = TrainSFT(model_id=model_id)
        sfttrainer.train(trainds, valds) # sft model trainer here, adapters will be saved in SFT_OUTPUT_DIR path

        # attach the sft adapter to original model, a new model will be stored which can be used for DPO fine tuning
        if os.path.exists(SFT_ADAPTER_DIR):
            model, processor = sfttrainer.load_model()
            attach_sft_adapter(model, SFT_ADAPTER_DIR, SFT_MODEL_DIR) # new model will be stored at SFT_MODEL_DIR

        dpotrainer = TrainDPO(model_id=model_id, prompt=prompt) # original model id to load processor
        dpotrainer.train(pref_data_csv_path=PREFERENCE_DATA_DPO)

        print("Model Trained with both methods")

    

    if infer:
        print("=====Inferencing the model =====")
        
        df = pd.read_csv(infer_data_csv)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        if not os.path.exists(SFT_MODEL_PATH):
                raise FileNotFoundError(
                    f"[ERROR] SFT Model does not exist. Make sure to attach adapters to the base model: {SFT_MODEL_PATH}"
                )
        
        results_df = pd.DataFrame()
        for ep in range(episodes):

            print("===== EPISODE 1 =====")
            # SFT model will be loaded
            model = AutoModelForVision2Seq.from_pretrained(
                SFT_MODEL_PATH,        
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                quantization_config=bnb_config,
                cache_dir = HF_HOME
            )

            processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)

            # attach DPO adapters to SFT model, or you can directly infer SFT model
            model.load_adapter(DPO_ADAPTERS_DIR)

            col_name = f'output_episode_{ep+1}'

            for i in range(len(df) - histlen - futrlen):
                inpimgs = df.loc[i: i+histlen-1, 'frame_path'].tolist()
                outputlabels = df.loc[i+histlen:i+histlen+futrlen-1, 'Action Label'].tolist()
                outputtext = get_model_output(inpimgs, model, processor, prompt=prompt, issegment=True)

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
        
        os.makedirs(ftoutputdir, exist_ok=True)
        csvpath = os.path.join(ftoutputdir, outputfilename)

        results_df.to_csv(csvpath, index=False)
        print("[FILED SAVED TO OUTPUTDIR] -> ", csvpath)
        
