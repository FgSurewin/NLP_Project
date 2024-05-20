import torch
import torch.nn as nn
from transformers import BertModel
from models.base_model import BaseModel


class EmotionClassifier(BaseModel):
    def __init__(self, num_classes, label_names, **kwargs):
        super(EmotionClassifier, self).__init__(num_classes, label_names)
        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, num_classes)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        logits = self.fc(pooled_output)
        return logits


if __name__ == "__main__":
    # Set global parameters
    NUM_CLASSES = 6

    # Create model
    model = EmotionClassifier(num_classes=NUM_CLASSES, label_names=None)

    # Create dummy input
    input_ids = torch.randint(0, 100, (64, 121), dtype=torch.long)
    mask = torch.ones((64, 121), dtype=torch.long)

    # Forward pass
    y = model(input_ids, mask)
    print(y.shape)
