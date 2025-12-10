import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import os
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "blocklens_model.pth"

class FeatureExtractor:
    def __init__(self):
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights).eval().to(DEVICE)
        self.model.classifier = nn.Identity()
        self.preprocess = weights.transforms()

    def get_features(self, image):
        image = image.convert('RGB')
        batch = self.preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = self.model(batch)
        return features.squeeze(0)

class BlockLensDistillationModel(nn.Module):
    def __init__(self, input_dim=576, num_supporting_signals=5):
        super(BlockLensDistillationModel, self).__init__()
        
        self.img_branch = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.signal_branch = nn.Sequential(
            nn.Linear(num_supporting_signals, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 3) 
        )

    def forward(self, img_features, signals):
        img_out = self.img_branch(img_features)
        sig_out = self.signal_branch(signals)
        combined = torch.cat((img_out, sig_out), dim=1)
        return self.classifier(combined)

class BlockLensManager:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.blocklens_model = BlockLensDistillationModel(num_supporting_signals=5).to(DEVICE)
        self.optimizer = optim.Adam(self.blocklens_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.labels_map = {"real_image": 0, "ai_generated": 1, "screenshot": 2}
        self.idx_to_label = {0: "real_image", 1: "ai_generated", 2: "screenshot"}
        
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.blocklens_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                print("Loaded existing BlockLens model.")
            except Exception as e:
                print(f"Failed to load BlockLens model: {e}")

    def save_model(self):
        torch.save(self.blocklens_model.state_dict(), MODEL_PATH)

    def predict(self, image, signals):
        self.blocklens_model.eval()
        img_features = self.extractor.get_features(image).unsqueeze(0)
        sig_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0).to(DEVICE) 
        
        with torch.no_grad():
            logits = self.blocklens_model(img_features, sig_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0][pred_idx].item() * 100
            
        return self.idx_to_label[pred_idx], confidence

    def train_step(self, image, signals, teacher_verdict):
        if teacher_verdict not in self.labels_map:
            return

        self.blocklens_model.train()
        
        img_features = self.extractor.get_features(image).unsqueeze(0)
        sig_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        target = torch.tensor([self.labels_map[teacher_verdict]], dtype=torch.long).to(DEVICE)

        self.optimizer.zero_grad()
        logits = self.blocklens_model(img_features, sig_tensor)
        loss = self.criterion(logits, target)
        
        loss.backward()
        self.optimizer.step()
        
        self.save_model()
        
        return loss.item()
