import torch
import torch.nn as nn
from transformers import BertModel
from models.base_model import BaseModel


class CNNEmotionClassifier(BaseModel):
    def __init__(self, num_classes, label_names, kernel_size=3, num_filters=256, dropout=0.2, **kwargs):
        super(CNNEmotionClassifier, self).__init__(num_classes, label_names)
        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        self.conv = nn.Conv1d(in_channels=768, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.fc = nn.Linear(num_filters, num_classes)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        # print(f"Before -> pooled_output: {pooled_output.shape}")
        pooled_output = pooled_output.unsqueeze(2)
        # print(f"After -> pooled_output: {pooled_output.shape}")
        conv_out = self.relu(self.conv(pooled_output))
        # print(f"conv_out: {conv_out.shape}")
        pooled_conv_out, _ = torch.max(conv_out, dim=2)
        # print(f"pooled_conv_out: {pooled_conv_out.shape}")
        
        pooled_conv_out = self.drop(pooled_conv_out)
        logits = self.fc(pooled_conv_out)
        return logits


if __name__ == "__main__":
    # Set global parameters
    NUM_CLASSES = 6

    # Create model
    model = CNNEmotionClassifier(num_classes=NUM_CLASSES, label_names=None)

    # Create dummy input
    input_ids = torch.randint(0, 100, (64, 121), dtype=torch.long)
    mask = torch.ones((64, 121), dtype=torch.long)

    # Forward pass
    y = model(input_ids, mask)
    print(y.shape)
