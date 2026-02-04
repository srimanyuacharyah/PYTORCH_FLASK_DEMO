import torch
import torch.nn as nn
import os

# Define the model architecture (adjust based on your model structure)
class TextGenerator(nn.Module):
    def __init__(self, vocab_size=5273, embedding_dim=16, hidden_dim=32, output_dim=5273, n_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(output))
        return predictions

def load_model(model_path):
    """Load the text generator model from .pth file"""
    try:
        # Initialize model with correct architecture
        model = TextGenerator(
            vocab_size=5273,           # From embedding.weight shape
            embedding_dim=16,          # From embedding.weight shape
            hidden_dim=32,             # From LSTM weight dimensions
            output_dim=5273,           # From fc.weight shape
            n_layers=1,                # Only 1 LSTM layer in checkpoint
            dropout=0.0                # Adjust as needed
        )
        
        # Load the model state dict directly
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        model.eval()  # Set to evaluation mode
        print("✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def generate_text(model, seed_text, num_predictions=50):
    """Generate text using the loaded model"""
    if model is None:
        print("Model not loaded. Cannot generate text.")
        return None
    
    try:
        # This is a placeholder for text generation logic
        # Adjust based on your specific model implementation
        model.eval()
        with torch.no_grad():
            # You would need to tokenize seed_text here
            # This is a simplified example
            generated_text = seed_text
            print(f"Generated text preview: {generated_text[:100]}...")
            return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return None

if __name__ == "__main__":
    model_path = "text generator.pth"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
    else:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        
        if model is not None:
            print("\nModel architecture:")
            print(model)
            print("\n✓ Model is ready for text generation!")
            
            # Example: Generate text with a seed
            seed_text = "The quick"
            generated = generate_text(model, seed_text, num_predictions=50)
