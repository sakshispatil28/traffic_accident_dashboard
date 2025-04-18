import torch
from model import Config, ObjectDetector, AccidentDetectionModel, generate_motion_features
from torchvision import transforms
import albumentations as A
from django.conf import settings
import os

config = Config()
config.DEVICE = torch.device("cpu")  # Use CPU for simplicity here

# Load the model
model = AccidentDetectionModel()
checkpoint = torch.load(os.path.join(settings.BASE_DIR, "best_model.pth"), map_location=config.DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

object_detector = ObjectDetector(device=config.DEVICE)

def predict_from_video(video_path):
    from model import VideoDataset
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoDataset([video_path], transform=transform, clip_len=config.VIDEO_FRAMES,
                           frame_size=config.FRAME_SIZE, mode='test')
    frames = dataset[0].unsqueeze(0)

    motion_features = generate_motion_features(frames, object_detector)
    with torch.no_grad():
        outputs = model(frames, motion_features)
        probs = torch.softmax(outputs, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0, 1].item()
