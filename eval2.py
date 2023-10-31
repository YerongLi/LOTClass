import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics and voting for predicted labels.")
    parser.add_argument("--vote", action="store_true", help="Perform voting from multiple output files")
    parser.add_argument("--model", choices=["vc", "llama", "lot"], default="lot", help="Choose the model")
    return parser.parse_args()

dataset = 'news'

# Read the ground truth labels
with open(f'datasets/{dataset}/{dataset}_train_labels.txt', 'r') as file:
    true_labels = [int(line.strip()) for line in file]

# Parse arguments
args = parse_args()

if args.vote:
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

else:
    # Default evaluation using 'lot' as the model
    model_result = {'vc': 'vc_out.txt', 'llama': 'llama_out.txt', 'lot': 'out.txt'}

    with open(f'datasets/{dataset}/{model_result}', 'r') as file:
        predicted_labels = [int(line.strip()) for line in file][:100]

    # Calculate precision, recall, and F1 score for the default model
    precision = precision_score(true_labels[:100], predicted_labels, average='macro')
    recall = recall_score(true_labels[:100], predicted_labels, average='macro')
    f1 = f1_score(true_labels[:100], predicted_labels, average='macro')

    print("Metrics for out.txt (default 'lot' model):")
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
