import torch
import os
import numpy as np
import pandas as pd
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

model_name = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name)

labels = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "a": 9,
    "b": 10, "c": 11, "d": 12, "e": 13, "f": 14, "g": 15, "h": 16, "i": 17, "j": 18,
    "k": 19, "l": 20, "m": 21, "n": 22, "o": 23, "p": 24, "q": 25, "r": 26, "s": 27,
    "t": 28, "u": 29, "v": 30, "w": 31, "x": 32, "y": 33, "z": 34
}

class CustomDataset(Dataset):
    def __init__(self, data_path, processor, labels):
        self.data_path = data_path
        self.processor = processor
        self.labels = labels
        self.image_paths = []
        self.targets = []

        self._prepare_dataset()

    def _prepare_dataset(self):
        for label in os.listdir(self.data_path):
            label_path = os.path.join(self.data_path, label)
            if not os.path.isdir(label_path):
                continue

            if label not in self.labels:
                print("Adding label", label)
                self.labels[label] = len(self.labels)

            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                self.image_paths.append(img_path)
                self.targets.append(self.labels[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()
        return {'pixel_values': pixel_values, 'labels': torch.tensor(label)}

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]),
    }

print("Starting data ingestion")
data_path = "asl_dataset"
dataset = CustomDataset(data_path, processor, labels)
# print(labels)

print("Finished processing data, starting training process")

# Split data into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

model = ViTForImageClassification.from_pretrained(model_name, num_labels=len(labels)).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',                  # output directory
    num_train_epochs=10,                      # total number of training epochs
    per_device_train_batch_size=16,          # batch size per device during training
    warmup_steps=500,                        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                       # strength of weight decay
    logging_dir='./logs',                    # directory for storing logs
    save_strategy="epoch",                   # Save model after each epoch
    eval_strategy="epoch",                   # Evaluate after each epoch
    load_best_model_at_end=True,             # Load the best model at the end of training
    metric_for_best_model="accuracy",        # Use accuracy to identify the best model
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                           # the instantiated 🤗 Transformers model to be trained
    args=training_args,                    # training arguments, defined above
    train_dataset=train_dataset,           # training dataset
    eval_dataset=test_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,       # function to compute metrics
    tokenizer=processor                    # passing the tokenizer for data preprocessing
)

# Start training
train_results = trainer.train()
print("Training completed")
eval_results = trainer.evaluate()
evaluation_percentage = eval_results['eval_accuracy'] * 100
print(f"Model evaluation accuracy: {evaluation_percentage:.2f}%")

# save the model 
model_path = "asl_model_austin"
model.save_pretrained(model_path)

# save the labels
labels_df = pd.DataFrame.from_dict(labels, orient='index', columns=['label'])
labels_df.to_csv('labels.csv')
print("Model and labels saved")