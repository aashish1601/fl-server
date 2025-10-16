#!/usr/bin/env python3
"""
Test and compare baseline vs final blended model
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net

def evaluate_model(model_path, model_name):
    """Evaluate a model on MNIST test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    # Load model
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    # Evaluate
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def main():
    print("\n" + "="*60)
    print("📊 COMPARING BASELINE vs BLENDED MODEL")
    print("="*60)
    
    try:
        baseline_acc = evaluate_model("baseline_model.pth", "Baseline Model (Server's initial knowledge)")
    except FileNotFoundError:
        print("❌ baseline_model.pth not found. Run 'python run_blend.py' first!")
        return
    
    try:
        final_acc = evaluate_model("models/final_blended_model.pth", "Final Blended Model (After FL)")
    except FileNotFoundError:
        print("❌ final_blended_model.pth not found. Run 'python run_blend.py' first!")
        return
    
    print("\n" + "="*60)
    print("📈 IMPROVEMENT SUMMARY")
    print("="*60)
    print(f"Baseline accuracy:  {baseline_acc:.2f}%")
    print(f"Final accuracy:     {final_acc:.2f}%")
    print(f"Improvement:        {final_acc - baseline_acc:+.2f}%")
    print("="*60)
    
    if final_acc > baseline_acc:
        print("✅ Server model improved through federated blending!")
    elif final_acc == baseline_acc:
        print("➖ No change (may need more rounds or different α)")
    else:
        print("⚠️  Accuracy decreased (try lowering α or check data quality)")
    print()

if __name__ == "__main__":
    main()


