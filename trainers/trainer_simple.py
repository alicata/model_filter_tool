import torch
import torch.nn as nn
import torch.optim as optim



# Logistic Regression Model
class LogisticRegressionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class AutoGeneratorTrainer:
    """
    AutoGeneratorTrainer generates samples internally and trains simple logistic regression binary classifier.
    
    Input: MxN float tensor (default 1x1 = 1d)
    Output: 1d float prediction
    
    """
    def __init__(self, input_dim=1):
        # Hyperparameters
        assert(input_dim==1) # 1d only supported for now
        self.input_dim = input_dim
        self.lr = 0.01
        self.num_epochs = 1000
        # Create an instance of the model
        self.model = LogisticRegressionNetwork(input_dim)
        self.data = self.gen_samples()
    
    def gen_samples(self, input_dim=1):
        # Generate some dummy data
        assert(input_dim==1) # 1d only supported for now
        # X is input features, y is binary labels
        X = torch.tensor([[0.1], [0.5], [0.9], [1.4], [1.8]], dtype=torch.float32)
        y = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32).view(-1, 1)
        return (X, y)

    def train(self):
        # Loss and Optimizer
        criterion = nn.BCELoss()  # Binary Cross-Entropy loss
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        X, y = self.data

        print(f"Train LogisticRegression model over {len(X)} samples")
        for X0, y0 in zip(X, y):
            # Training Loop
            for epoch in range(self.num_epochs):
                # Forward pass
                outputs = self.model(X)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (epoch+1) % 100 == 0:
                    print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')
        return self

    def evaluate(self):
        X, y = self.data

        # Make predictions on the training data
        with torch.no_grad():
            predictions = self.model(X)
            predictions = (predictions > 0.5).float()
            accuracy = (predictions == y).float().mean()
            print(f'Accuracy: {accuracy * 100:.2f}%')
        return self
    
    def save(self, filepath='./model_auto_generated.pth'):
        torch.save(self.model.state_dict(), filepath)
        return self
    
if __name__ == "__main__":
    AutoGeneratorTrainer().train().evaluate().save('model_unit_test.pth')


