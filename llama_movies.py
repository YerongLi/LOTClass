import os
from math import ceil
import logging
import torch
import os
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM

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
model_path = "gpt2"
model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/"
# device = "cuda:0"  # You can set this to "cpu" if you don't have a GPU
device='cuda:0'

# Load the model
logging.info('Loading the model')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # device_map='auto',
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
predicted_labels = []

text_file='movies_train.txt'
dataset_dir='datasets/movies'
loader_file='llama_movies.pt'
print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
docs = [doc.strip() for doc in corpus.readlines()]
docs = docs[:120]
print(f"Converting texts into tensors.")
prompt = """
In the following movie review classification task, you are given a text.
Your goal is to classify whether the movie review is 'bad' or 'good'.
Please use the text provided to predict the sentiment of the movie review.
Choose between 'bad' and 'good' as the label for classification.

Input: Every great gangster movie has under-currents of human drama. Don't expect an emotional story of guilt, retribution and despair from "Scarface". This is a tale of ferocious greed, corruption, and power. The darker side of the fabled "American Dream". Anybody complaining about the "cheesiness" of this film is missing the point. The superficial characters, cheesy music, and dated fashions further fuel the criticism of this life of diabolical excess. Nothing in the lives of these characters really matter, not on any human level at least. In fact the film practically borderlines satire, ironic considering all the gangsta rappers that were positively inspired by the lifestyle of Tony Montana. This isn't Brian DePalma's strongest directorial effort, it is occasionally excellent and well-handled ( particularly the memorable finale ) , but frequently sinks to sloppy and misled. Thankfully, it is supported by a very strong script by Oliver Stone ( probably good therapy for him, considering the coke habit he was tackling at the time ) . The themes are consistent, with the focus primarily on the life of Tony Montana, and the evolution of his character as he is consumed by greed and power. The dialogue is also excellent, see-sawing comfortably between humour and drama. There are many stand-out lines, which have since wormed their way into popular culture in one form or another. The cast help make it what it is as well, but this is really Pacino's film. One of his earlier less subtle performances ( something much more common from him nowadays ) , this is a world entirely separate from Michael Corleone and Frank Serpico. Yet he is as watchable here as ever, in very entertaining ( and intentionally over-the-top ) form. It is hard to imagine another Tony Montana after seeing this film, in possibly one of the most mimicked performances ever. Pfeiffer stood out as dull and uncomfortable on first viewing, but I've come to realize how she plays out the part of the bored little wife. Not an exceptional effort, but unfairly misjudged. The supporting players are very good too, particularly Paul Shenar as the suave Alejandro Sosa. Powerful, occasionally humorous, sometimes shocking, and continually controversial. "Scarface" is one of the films of the eighties ( whatever that might mean to you ) . An essential and accessible gangster flick, and a pop-culture landmark. 9/10
Output: good

Input : I chuckled a few times during this movie. I laughed out loud during the notarizing of the margarine company handover ( pun intended ) . There are three segments in this movie. The first one is supposed to be a spoof of "woman 'grows up' and launches career" movies. The Tampax_ box was the funniest thing in this segment. Most of the cast members aren't listed here on IMDb. They are the lucky ones. Few other people will be able to connect this thing to the ruin of their acting careers. The second segment is a spoof of "sharkish woman sleeps her way to the top and seizes control of huge industry" movies. Robert Culp has several funny moments, all physical humor, including the aforementioned handover. After his character dies the segment sinks lower and lower as Dominique Corsaire rises higher and higher. By the time she becomes First Lady I wanted to rip the cable out of the TV and watch "snow." I switched to Pakistani music videos instead. I don't understand Urdu, or whatever language the videos were in. It was still better than listening to the dialogue in this painfully dull "story." Then came "Municipalians" with the *big* stars, half of them on screen for less than a minute: Elisha Cook, Jr., Christopher Lloyd, Rhea Perlman, Henny Youngman, Julie Kavner, Richard Widmark and ... *Robby Benson.* It's supposed to be a spoof of "young cop teams with hardened, substance abusing older cop who needs retirement *badly*" movies. The horizontal flash bar on the police car is very impressive. It was interesting seeing old RTD buses, and a Shell gas station sign, and an American Savings sign -- none of them are around anymore. Nagurski's "Never stop anywhere you might have to get out the car" made me smile momentarily. Then they discuss how boring the young cop is. A lot. Back and forth about how boring he is. That was as boring as this description of how boring it is. Nagurski's Law Number Four, "Never go into a music store that's been cut into with an acetylene torch," made me think that the music store is a real business at the actual location the dispatcher gave. Thinking about that was more interesting than the set-up for the gag which followed. Young Falcone ( Benson ) gets shot. A lot. He becomes a hardened cop like Nagurski. The segment keeps going. On and on. And on. It won't stop. It rolls relentlessly onward no matter how many times you wish he'd just *die* already so this thing will end. It doesn't. It goes on and on and on.... Then a "Buffy the Vampire Slayer" episode which I've seen four times already comes on. Thank God! This abysmal movie ended while I went to get the mail.
Output: bad

Input: Let me start off by saying that this doesn't seem or feel like a movie. It seems like just another TV show about popular girls and boys with no real film language top back it up. The camera angles are so straight forward that the story is told the simplest way possible never making the public connect with it. This film takes us to where no movie I've seen has done before: to a realm where the Film Theater becomes a warm medium giving the public every element of interpretation. Too obvious. The large movie screen is only used as an enlarged TV from where we can see every attribute of these women in a larger than life manner. Lately it seems that young directors are compromising the Art of film making for sales. This is very scary. The industry is spoiling the art in movies. We must educate ourselves and our children about what cinematography and its language are really about. Not just sales and entertainment, but a way to communicate feelings, passions and even culture. Not as a launching platform for young divas and jocks.
Output: bad

Input: Well, if you are one of those Katana's film-nuts ( just like me ) you sure will appreciate this metaphysical Katana swinging blood spitting samurai action flick. Starring Tadanobu Asano ( Vital, Barren Illusion ) & Ryu Daisuke ( Kagemusha ) . This samurai war between Heiki's clan versus Genji's clan touch the zenith in the final showdown at Gojo bridge. The body-count is countless. Demons, magic swords, Shinto priests versus Buddhist monks and the beautiful visions provided by maestro Sogo Ishii will do the rest. A good Japanese flick for a rainy summer night.
Output: good

Input: """
output_map = {"bad": 0, "good": 1}
for input_text in tqdm(docs):
    # Convert text to input_ids
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Get the model output for the input_ids
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs[0]

    # Extract the probability for the token "good" and "bad"
    output_map = {"bad": 0, "good": 1}
    probs = torch.softmax(logits, dim=-1)
    prob_good = probs[0, -1, tokenizer.encode("good")[0]]
    prob_bad = probs[0, -1, tokenizer.encode("bad")[0]]

    # Map the probabilities to labels
    predicted_label = 1 if prob_good > prob_bad else 0

    # Append the predicted label to the list
    predicted_labels.append(predicted_label)

# Write the predicted labels to the "llama_out.txt" file
with open("llama_out.txt", "w") as file:
    for label in predicted_labels:
        file.write(str(label) + "\n")
    
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