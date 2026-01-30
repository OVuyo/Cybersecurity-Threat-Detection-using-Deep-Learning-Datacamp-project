"""
Main Script for Training Cybersecurity Threat Detection Model

This script demonstrates the complete workflow from data loading
to model training and evaluation.
"""

import torch
from src import (
    create_model,
    load_and_prepare_data,
    train_model,
    full_evaluation,
    save_model,
    print_data_info,
    print_evaluation_results
)


def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    TRAIN_PATH = 'data/labelled_train.csv'
    TEST_PATH = 'data/labelled_test.csv'
    VAL_PATH = 'data/labelled_validation.csv'
    
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = 'models/cybersecurity_model.pth'
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    data = load_and_prepare_data(
        TRAIN_PATH, 
        TEST_PATH, 
        VAL_PATH, 
        batch_size=BATCH_SIZE
    )
    print_data_info(data)
    
    # Create model
    print("\nCreating model...")
    input_size = data['X_train'].shape[1]
    model = create_model(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=2
    )
    print(f"Model created with {input_size} input features and {HIDDEN_SIZE} hidden neurons\n")
    
    # Train model
    print("Training model...")
    print("-" * 50)
    history = train_model(
        model,
        data['train_loader'],
        data['val_loader'],
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        verbose=True
    )
    print("-" * 50)
    
    # Evaluate on all datasets
    print("\nEvaluating model on all datasets...")
    results = full_evaluation(
        model,
        data['train_loader'],
        data['val_loader'],
        data['test_loader'],
        device=device
    )
    print_evaluation_results(results)
    
    # Save model
    print("Saving model...")
    save_model(model, MODEL_SAVE_PATH)
    
    print("\n✓ Training completed successfully!")
    print(f"Final Test Accuracy: {results['test_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
