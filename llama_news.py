import os
from math import ceil
import logging
import torch
import os
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM
dataset='news'
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
# model_path, model_name = "/scratch/yerong/.cache/pyllama/vicuna-7b-v1.3", "vc"
model_path, model_name = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/", 'llama'
print(model_name)
print("Ending")
# device = "cuda:0"  # You can set this to "cpu" if you don't have a GPU
device='cuda:0'

# Load the model
logging.info('Loading the model')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto',
).eval()

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
# ).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
predicted_labels = []
predicted_tokens = []
text_file=f'{dataset}_train.txt'
dataset_dir=f'datasets/{dataset}'
# loader_file='llama_{dataset}}.pt'
print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
docs = [doc.strip() for doc in corpus.readlines()]
docs = docs[:120]
print(f"Converting texts into tensors.")
if model_name =='llama':
    prompt = """
For the text classification task, you will be provided with textual information on various subjects: politics, sports, business, and technology.
Please utilize the content provided to classify the text into one of the following topics: 'politics', 'sports', 'business', or 'technology'.
Choose the category that best represents the content of the text.

Input: Marino Doesn't See Return to Dolphins Soon ( AP ) . AP - Dan Marino misses being part of the Miami Dolphins, yet does not see a scenario where he'd soon consider returning to the team's front office.
Output: sports

Input : Office Depot chairman resigns. Bruce Nelson, chairman and chief executive of Office Depot Inc. of Delray Beach, Fla., Monday resigned by mutual agreement with the board.
Output: business

Input: Apple Extends iTunes to Europe. The EU iTunes Music Store retains the same features and per-song price of 99 euro cents, established in June for customers in UK, Germany and France.
Output: technology

Input: Karzai #39;s lead shows Afghan ethnic divide. Hamid Karzai has been cruising to victory in Afghanistan #39;s first direct presidential elections, but the returns so far have underscored the ethnic fault lines that often divide the country.
Output: politics

Input: Ukraine PM on Verge of Victory, Rival Cries Foul. KIEV ( Reuters ) - Ukraine's prime minister was on the verge of victory in a presidential election on Monday but his liberal rival accused the authorities of mass fraud and told thousands of supporters to stay on the streets in protest.
Output: politics

Input: FDA approves injecting ID chips in patients. The US Food and Drug Administration has approved the practice of injecting humans with tracking devices for medical purposes, according to a Florida company that makes the devices.
Ouput: business

Input: """
else:
    prompt = """
For the text classification task, you will be provided with textual information on various subjects: politics, sports, business, and technology.
Please utilize the content provided to classify the text into one of the following topics: 'politics', 'sports', 'business', or 'technology'.
Choose the category that best represents the content of the text.

Input: Marino Doesn't See Return to Dolphins Soon ( AP ) . AP - Dan Marino misses being part of the Miami Dolphins, yet does not see a scenario where he'd soon consider returning to the team's front office.
Output: sports

Input : Office Depot chairman resigns. Bruce Nelson, chairman and chief executive of Office Depot Inc. of Delray Beach, Fla., Monday resigned by mutual agreement with the board.
Output: business

Input: Apple Extends iTunes to Europe. The EU iTunes Music Store retains the same features and per-song price of 99 euro cents, established in June for customers in UK, Germany and France.
Output: technology

Input: Karzai #39;s lead shows Afghan ethnic divide. Hamid Karzai has been cruising to victory in Afghanistan #39;s first direct presidential elections, but the returns so far have underscored the ethnic fault lines that often divide the country.
Output: politics

Input: Ukraine PM on Verge of Victory, Rival Cries Foul. KIEV ( Reuters ) - Ukraine's prime minister was on the verge of victory in a presidential election on Monday but his liberal rival accused the authorities of mass fraud and told thousands of supporters to stay on the streets in protest.
Output: politics

Input: FDA approves injecting ID chips in patients. The US Food and Drug Administration has approved the practice of injecting humans with tracking devices for medical purposes, according to a Florida company that makes the devices.
Ouput: business

Input: """
output_map = {"politics": 0, "sports": 1, "business" : 2, "technology": 4, None: 0}
for input_text in tqdm(docs):
    # Convert text to input_ids
    input_ids = tokenizer.encode(prompt+input_text+"\nOutput:", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=input_ids.shape[1] + 1)
        predicted_token_id = outputs[:, -1]

    # Decode the predicted token
    predicted_token = tokenizer.decode(predicted_token_id[0].item())
    print(predicted_token)
    try:
        predicted_label = output_map[predicted_token]
    except :
        predicted_label = 0
    # Append the predicted token to the list
    predicted_tokens.append(predicted_token)
    predicted_labels.append(predicted_label)
# Write the predicted tokens to the "llama_out.txt" file
# with open("llama_out.txt", "w") as file:
#     for token in predicted_tokens:
#         file.write(token + "\n")


with open(os.path.join(dataset_dir,f"{model_name}_out.txt"), "w") as file:
    for label in predicted_labels:
        file.write(str(label) + "\n")



    # # Get the model output for the input_ids
    # with torch.no_grad():
    #     outputs = model(input_ids=input_ids)
    #     logits = outputs[0]
    #     print(logits.shape)

    # # Extract the probability for the token "good" and "bad"
    # output_map = {"bad": 0, "good": 1}
    # probs = torch.softmax(logits, dim=-1)
    # prob_good = probs[0, -1, tokenizer.encode("good")[0]]
    # prob_bad = probs[0, -1, tokenizer.encode("bad")[0]]

    # # Map the probabilities to labels
    # predicted_label = 1 if prob_good > prob_bad else 0

    # # Append the predicted label to the list
    # predicted_labels.append(predicted_label)

# Write the predicted labels to the "llama_out.txt" file
# with open("llama_out.txt", "w") as file:
    # for label in predicted_labels:
        # file.write(str(label) + "\n")
    
# chunk_size = 4
# # chunk_size = ceil(len(docs) / self.num_cpus)
# chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
# # results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
# input_ids = torch.cat([result[0] for result in results])
# attention_masks = torch.cat([result[1] for result in results])
# print(f"Saving encoded texts into {loader_file}")
# if label_file is not None:
#     print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
#     truth = open(os.path.join(dataset_dir, label_file))
#     labels = [int(label.strip()) for label in truth.readlines()]
#     labels = torch.tensor(labels)
#     data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
# else:
#     data = {"input_ids": input_ids, "attention_masks": attention_masks}
# torch.save(data, loader_file)