
from src.data_loader import load_data
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model
import torch
import os

def main():
    print("Starting CNN CIFAR-10 Project (PyTorch)...")
    
    # Load Data
    print("Loading data...")
    trainloader, testloader, classes = load_data()
    print(f"Data loaded: Train batches {len(trainloader)}, Test batches {len(testloader)}")

    # Create Model
    print("Creating model...")
    model = create_model()
    # print(model)

    # Train Model
    print("Training model...")
    # Using 1 epoch for verification
    history = train_model(model, trainloader, testloader, epochs=10) 

    # Evaluate Model
    print("Evaluating model...")
    accuracy = evaluate_model(model, testloader)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/cifar10_cnn.pth')
    print("Model saved to models/cifar10_cnn.pth")

if __name__ == "__main__":
    main()
