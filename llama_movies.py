import os
from math import ceil
import logging
import torch
import os
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM
from modeling_llama_partial import LlamaForCausalLMPartial

logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S',
    force=True)
def todevice(feat, device):
    # Initialize an empty list to store the modified tensors
    modified_feat = []

    # Iterate over items in feat
    for item in feat:
        if isinstance(item, torch.Tensor):
            # If the item is a torch tensor, cast it to the specified device
            modified_item = item.to(device)
            modified_feat.append(modified_item)
        else:
            # If the item is not a tensor, leave it as it is
            modified_feat.append(item)

    # Return the modified feat
    return tuple(modified_feat)
logging.info(f'Logger start: {os.uname()[1]}')
# model_path = "/scratch/yerong/.cache/pyllama/vicuna-7b-v1.3"
# model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-chat-hf/"
model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-chat-hf/"
device = "cuda:0"  # You can set this to "cpu" if you don't have a GPU

# Load the model
logging.info('Loading the model')
model = LlamaForCausalLMPartial.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
).eval()
text_file='movies_train.txt'
dataset_dir='datasets/movies'
loader_file='llama.pt'
print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
docs = [doc.strip() for doc in corpus.readlines()]
print(f"Converting texts into tensors.")
chunk_size = ceil(len(docs) / self.num_cpus)
chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
input_ids = torch.cat([result[0] for result in results])
attention_masks = torch.cat([result[1] for result in results])
print(f"Saving encoded texts into {loader_file}")
if label_file is not None:
    print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
    truth = open(os.path.join(dataset_dir, label_file))
    labels = [int(label.strip()) for label in truth.readlines()]
    labels = torch.tensor(labels)
    data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
else:
    data = {"input_ids": input_ids, "attention_masks": attention_masks}
torch.save(data, loader_file)