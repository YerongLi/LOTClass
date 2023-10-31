import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics and voting for predicted labels.")
    parser.add_argument("--vote", action="store_true", help="Perform voting from multiple output files")
    parser.add_argument("--model", choices=["vc", "llama", "lot"], default="lot", help="Choose the model")
    parser.add_argument("--data", choices=["movies", "news"], default="movies", help="Choose the model")
    return parser.parse_args()

# Parse arguments
args = parse_args()
dataset = arg.data
# Read the ground truth labels
with open(f'datasets/{dataset}/{dataset}_train_labels.txt', 'r') as file:
    true_labels = [int(line.strip()) for line in file]


if args.vote:
    filenames = ['out.txt', 'vc_out.txt', 'llama_out.txt']
    predicted_labels = []

    for filename in filenames:
        with open(f'datasets/{dataset}/{filename}', 'r') as file:
            labels = [int(line.strip()) for line in file]
            predicted_labels.append(labels[:100])  # Considering 100 labels from each file for voting

    # Voting among the predicted labels
    voted_predictions = [max(set(prediction), key=prediction.count) for prediction in zip(*predicted_labels)]

    # Calculate precision, recall, and F1 score based on voting
    precision = precision_score(true_labels[:100], voted_predictions, average='macro')
    recall = recall_score(true_labels[:100], voted_predictions, average='macro')
    f1 = f1_score(true_labels[:100], voted_predictions, average='macro')

    print("Metrics based on voting:")
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

else:
    model_filenames = {'vc': 'vc_out.txt', 'llama': 'llama_out.txt', 'lot': 'out.txt'}
    model_file = model_filenames[args.model]

    # Read the predicted labels based on the chosen model
    with open(f'datasets/{dataset}/{model_file}', 'r') as file:
        predicted_labels = [int(line.strip()) for line in file][:100]

    # Calculate precision, recall, and F1 score for the selected model
    precision = precision_score(true_labels[:100], predicted_labels, average='macro')
    recall = recall_score(true_labels[:100], predicted_labels, average='macro')
    f1 = f1_score(true_labels[:100], predicted_labels, average='macro')

    print(f"Metrics for {model_file}:")
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
