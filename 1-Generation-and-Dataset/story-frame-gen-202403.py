# Script to load a specified chat model, and ask it for a set of stories that invoke certain frames.
# Author: Hadi
# Date: v20240312

# prereq: pip install -U transformers accelerate sentencepiece protobuf bitsandbytes pytorch
 
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

REPEATS = 1
FRAMES = ['strict father', 'nurturing parent',
          'us vs. them',  'we are all in this together',
          'nature cannot be controlled', 'mastery over nature',
          'illusions to enlightenment', "society of the spectacle"]
          # better not to use overlapping frames such as 'utopia'
SOURCES = ['orig', 'bible', 'scifi']


# Get command line arguments for script
parser = argparse.ArgumentParser(description='Story by frame script')
parser.add_argument('--model', type=str, required=True, help='Name of the model (llama2, yi, vicuna, mistral)')
parser.add_argument('--output', type=str, required=True, help='Name of the output CSV file')
args = parser.parse_args()

filename = args.output  if args.output.endswith('csv') else args.output + '.csv'
file_exists = os.path.isfile(filename)
outfile = open(filename, 'a')
if not file_exists:
    outfile.write("MODEL\tFRAME\tSOURCE\tSTORY\tEVAL\n")

hf_model = {"llama2": "meta-llama/Llama-2-7b-chat-hf",
            "llama2-q": "TheBloke/Llama-2-7B-Chat-GPTQ",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.2",  # likes to have pad_token_id?
            "mistral-q": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
            "yi": "01-ai/Yi-6B-Chat",
            "yi-q": "TheBloke/Yi-6B-GPTQ",
            "vicuna": "lmsys/vicuna-7b-v1.5",  # was answering blank: add 'QA' helped! (and perhaps more tokens?) @TODO thats the reason to archive Y story_frame.py
            "vicuna-q": "TheBloke/vicuna-7B-v1.5-GPTQ"
            }[args.model]

print("LOADING MODEL:", hf_model)
tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, device_map="auto")  # On GPU (FP32 or quantized)
#model.cuda()  # ON GPU, FP32 (no quant)


st = time()
for frame in FRAMES:
    print("\n****", frame, "****")
    for source in SOURCES:
        if source=='orig':
            theq = 'Question: please write a short original story which invokes the "' + frame + '" frame (in one paragraph).\nAnswer: '
        elif source=='bible':
            theq = 'Question: please pick a short passage from the Bible which invokes the "' + frame + '" frame (in one paragraph).\nAnswer: '  
        elif source=='scifi':
            theq = 'Question: please describe a short story from a scifi novel or movie which invokes the "' + frame + '" frame (in one paragraph).\nAnswer: '
        for repeat in range(REPEATS):
            input_ids = tokenizer(theq, return_tensors='pt')['input_ids'].cuda()
            output_ids = model.generate(input_ids, max_new_tokens=300, temperature=0.7, do_sample=True)  # (maybe also top_p? )  
            thestory = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            thestory = thestory.replace(theq, "").replace("\n---\n", "").replace("\n", " ").replace('\t', ' ').strip()
            print("@", round(time()-st, 1), "s → ", source, len(output_ids[0]), "toks :")
            outfile.write(args.model + "\t" + frame.replace(' ','-') + '\t' + source + '\t' + thestory + '\t\n')
            
 
            print(thestory)


