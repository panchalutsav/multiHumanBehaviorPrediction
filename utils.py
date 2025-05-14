from prompts import predprompt
from PIL import Image
import yaml
import os
import pandas as pd
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import torch

with open('configs/general.yaml', 'r') as f:
    gconfig = yaml.safe_load(f)

imgbasepath = gconfig['imgbasepath']
MAX_NEW_TOKENS = gconfig['MAX_NEW_TOKENS']
TEMPERATURE = gconfig['TEMPERATURE']

def buildprompt(scenegraph, availableactions, histlen, futrlen):
    # predprompt.format()
    # dpoBasePrompt.format()
    
    return predprompt.format(
        histlen=histlen, 
        futrlen=futrlen, 
        # scenedesc=scenedesc, 
        scenegraph=scenegraph, 
        availableactions=availableactions
    )

def qwenVLDataFormat(history_images, future_labels, query):
    # there are two ways of data format, include system message and not include system message
    return {"messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": [Image.open(os.path.join(imgbasepath, frame)) for frame in history_images],
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



def attach_sft_adapter(model, sft_adapter_path, model_output_path):
    """
    model_output_path : here the new model will be saved after the adapter is attached
    """
    peftmodel = PeftModel.from_pretrained(model, sft_adapter_path)
    merged_model = peftmodel.merge_and_unload()
    merged_model.save_pretrained(model_output_path, safe_serialization=True, max_shard_size="2GB")
    print("Model merged and saved")
    del model 
    return 1




def get_model_output(imgpaths, model, processor, prompt,issegment=False):
    assert isinstance(imgpaths, list) == True, f"imgpaths should be of 'list' type, given {type(imgpaths)}"
    conversation = [
            # { "role": "system","content": [{"type": "text", "text": system_message}],
            # },
            {"role": "user", "content": [
                    {
                        "type": "video",
                        "video": [Image.open(os.path.join(imgbasepath, frame)) for frame in imgpaths],
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
    ]

    text_prompt = processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)

    if issegment: # workaround for problem with huggingface to use list of vectors as image inputs
        videoinputsX = [videoinput[0] for videoinput in video_inputs]
        inputs = processor(text=[text_prompt], videos=videoinputsX, padding=True, return_tensors="pt").to('cuda')
    else:
        inputs = processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, top_p=1.0, do_sample=True, temperature=TEMPERATURE)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]


