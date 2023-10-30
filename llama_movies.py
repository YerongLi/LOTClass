import os
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