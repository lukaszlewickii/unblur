import time
import numpy as np
import pandas as pd
import cv2
import psutil
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable
import statistics
import torchvision.models as models
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

@dataclass
class BenchmarkResult:
    model_name: str
    hardware: str
    avg_latency_ms: float
    std_latency_ms: float
    throughput_imgs_sec: float
    memory_mb: float
    accuracy: float = None
    
def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def benchmark_model(
    model_fn: Callable,
    image_paths: List[Path],
    name: str,
    warmup_runs: int = 5,
    test_runs: int = 100,
    batch_size: int = 1
) -> BenchmarkResult: 
    """
    Rzetelny benchmark pojedynczego modelu
    
    Args:
        model_fn: Function that takes image(s) and returns prediction
        image_paths: List of test images
        name: Model name
        warmup_runs:  Runs to discard (cache warming)
        test_runs: Actual measurement runs
        batch_size: Images per batch
    """
    
    images = [cv2.imread(str(p)) for p in image_paths[: test_runs]]
    
    print(f"Warming up {name}...")
    for i in range(warmup_runs):
        _ = model_fn(images[i % len(images)])
    
    print(f"Benchmarking {name} ({test_runs} runs)...")
    latencies = []
    mem_before = measure_memory()
    
    for i in range(0, test_runs, batch_size):
        batch = images[i:i+batch_size]
        
        start = time.perf_counter()
        _ = model_fn(batch[0] if batch_size == 1 else batch)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000 / len(batch)
        latencies.append(latency_ms)
    
    mem_after = measure_memory()
    
    avg_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies)
    throughput = 1000 / avg_latency
    memory_used = mem_after - mem_before
    
    hardware = "CPU"
    if torch.cuda.is_available():
        hardware = f"GPU ({torch.cuda.get_device_name(0)})"
    
    return BenchmarkResult(
        model_name=name,
        hardware=hardware,
        avg_latency_ms=avg_latency,
        std_latency_ms=std_latency,
        throughput_imgs_sec=throughput,
        memory_mb=memory_used
    )

def classical_ml_model(image):
    """OpenCV + XGBoost"""
    if isinstance(image, list):
        image = image[0]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    max_dim = 800
    h, w = gray.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.mean(np.sqrt(gx**2 + gy**2))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    contrast = gray.std()
    
    features = [laplacian_var, gradient_mag, edge_density, contrast]
    
    # Mock XGBoost prediction (replace with your actual model)
    # prediction = xgb_model. predict([features])
    prediction = features[0] > 100  # Mock threshold
    
    return prediction

# Small CNN (MobileNetV3 + custom head)

class BlurDetectorCNN: 
    def __init__(self, device='cpu'):
        self.device = device
        # Lightweight model
        self.model = models.mobilenet_v3_small(weights='DEFAULT')
        # Replace classifier for binary classification
        self.model.classifier[-1] = torch.nn.Linear(
            self.model.classifier[-1].in_features, 2
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        if isinstance(image, list):
            image = image[0]
        
        # Preprocess
        img_tensor = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction

# Vision Language Model (CLIP-based)

class VLMBlurDetector:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(device)
        self.model.eval()
        
        self.text_inputs = self.processor(
            text=["a blurry photo", "a sharp photo"],
            return_tensors="pt",
            padding=True
        ).to(device)
    
    def __call__(self, image):
        if isinstance(image, list):
            image = image[0]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        inputs = self.processor(
            images=image_rgb,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            text_features = self.model.get_text_features(**self.text_inputs)
            
            # Cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            prediction = similarity[0][0] > similarity[0][1]  # blurry > sharp? 
        
        return prediction. item()

# Specialized blur detection CNN (lightweight)
class TinyBlurNet:
    """Custom tiny CNN for blur detection"""
    def __init__(self, device='cpu'):
        self.device = device
        self.model = torch. nn.Sequential(
            # Input:  3x224x224
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x112x112
            torch.nn. ReLU(),
            torch. nn.MaxPool2d(2),  # 32x56x56
            
            torch.nn.Conv2d(32, 64, 3, padding=1),  # 64x56x56
            torch.nn. ReLU(),
            torch. nn.MaxPool2d(2),  # 64x28x28
            
            torch.nn.Conv2d(64, 128, 3, padding=1),  # 128x28x28
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),  # 128x1x1
            
            torch. nn. Flatten(),
            torch.nn. Linear(128, 2)
        ).to(device)
        self.model.eval()
        
        self.transform = transforms. Compose([
            transforms.ToPILImage(),
            transforms. Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def __call__(self, image):
        if isinstance(image, list):
            image = image[0]
        
        img_tensor = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction

def run_all_benchmarks(image_dir: Path, output_csv: str = "benchmark_results. csv"):
    """Run comprehensive benchmark suite"""
    
    image_paths = list(image_dir.glob("*.jpg"))[:100]
    
    results = []
    
    print("\n" + "="*50)
    print("CPU BENCHMARKS")
    print("="*50 + "\n")
    
    models = [
        (classical_ml_model, "Classical ML (OpenCV + XGBoost)"),
        (TinyBlurNet(device='cpu'), "TinyBlurNet (Custom CNN)"),
        (BlurDetectorCNN(device='cpu'), "MobileNetV3 Small"),
    ]
    
    for model_fn, name in models:
        result = benchmark_model(model_fn, image_paths, name, test_runs=100)
        results.append(result)
        print(f"\n{name}:")
        print(f"  Latency: {result.avg_latency_ms:.2f} ± {result.std_latency_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_imgs_sec:.1f} images/sec")
        print(f"  Memory: {result.memory_mb:.1f} MB")
    
    if torch.cuda. is_available():
        print("\n" + "="*50)
        print("GPU BENCHMARKS")
        print("="*50 + "\n")
        
        gpu_models = [
            (TinyBlurNet(device='cuda'), "TinyBlurNet (Custom CNN)"),
            (BlurDetectorCNN(device='cuda'), "MobileNetV3 Small"),
            (VLMBlurDetector(device='cuda'), "CLIP VLM"),
        ]
        
        for model_fn, name in gpu_models:
            result = benchmark_model(model_fn, image_paths, name + " [GPU]", test_runs=100)
            results.append(result)
            print(f"\n{name}:")
            print(f"  Latency: {result.avg_latency_ms:.2f} ± {result.std_latency_ms:.2f} ms")
            print(f"  Throughput: {result. throughput_imgs_sec:.1f} images/sec")
            print(f"  Memory: {result.memory_mb:.1f} MB")
    
    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv(output_csv, index=False)
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks(Path("path/to/test/images"))