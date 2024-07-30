---
slug: adding-learner
title: Adding Learner
tags: [math-learner]
---

In this post, I'm going to describe building a simple neural network using PyTorch that learns to add two numbers.

### Data Generation

First, I'll define a method that generates the data. This method returns two tensors. The first tensor, `x`, is the input to the model, and the second tensor, `y`, is the expected output (the sum of the pairs of numbers).

```python
def generate_data(num_samples=1000):
    x = torch.randint(0, 100, (num_samples, 2), dtype=torch.float32)
    y = torch.sum(x, dim=1, keepdim=True)
    return x, y
```
<!--truncate-->

### Defining the Model

Next, I'll define a simple neural network module. This module has two linear layers with a ReLU activation function between them. The first layer takes two inputs, and the second layer outputs one value.

```python
class SimpleAdder(nn.Module):
    def __init__(self):
        super(SimpleAdder, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Training the Model

Here's the method to train the model. Every 100 epochs, the training loss is printed to track progress.

```python
def train_model(model, criterion, optimizer, x_train, y_train, num_epochs=1000):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### Testing the Model and Calculating Accuracy

Next, I'll define a couple of methods to test the model and calculate accuracy. The model returns a number with a fractional part, which is useful for training because it gives the loss function a smoother gradient. When actually computing accuracy, I round the predictions to compare them with the whole number answers.

```python
def test_model(model, x_test):
    with torch.no_grad():
        predicted = model(x_test)
        rounded_predicted = torch.round(predicted)
    return rounded_predicted

def calculate_accuracy(predictions, targets):
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total * 100
    return correct, total, accuracy
```

## Main Function

Here's the main function that puts everything together.

```python
# Generate training and test data
x_train, y_train = generate_data(1000)
x_test, y_test = generate_data(10)
    
# Initialize the model, criterion, and optimizer
model = SimpleAdder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
    
# Train the model
train_model(model, criterion, optimizer, x_train, y_train)
    
# Test the model
y_pred = test_model(model, x_test)
    
# Calculate accuracy
correct, total, accuracy = calculate_accuracy(y_pred, y_test)
    
# Print results
print(f'Predictions:\n{y_pred}')
print(f'Actual sums:\n{y_test}')
print(f'Number of exactly correct predictions: {correct}/{total}')
print(f'Accuracy: {accuracy:.2f}%')
```

### Output and Results

Here is the output. The neural network shows that it is learning by the decreasing loss values.

```
Epoch [100/1000], Loss: 2.5868
Epoch [200/1000], Loss: 0.3593
Epoch [300/1000], Loss: 0.2960
Epoch [400/1000], Loss: 0.2394
Epoch [500/1000], Loss: 0.1914
Epoch [600/1000], Loss: 0.1523
Epoch [700/1000], Loss: 0.1209
Epoch [800/1000], Loss: 0.0965
Epoch [900/1000], Loss: 0.0771
Epoch [1000/1000], Loss: 0.0615
```

After the model has finished training, here are the results of running it on the test set.

```
Predictions:
tensor([[ 64.],
        [131.],
        [114.],
        [ 82.],
        [ 86.],
        [ 47.],
        [ 76.],
        [ 32.],
        [112.],
        [ 17.]])

Actual sums:
tensor([[ 64.],
        [131.],
        [114.],
        [ 82.],
        [ 86.],
        [ 47.],
        [ 76.],
        [ 32.],
        [112.],
        [ 17.]])

Number of exactly correct predictions: 10/10
Accuracy: 100.00%
```

### Wrap-up

This simple neural network successfully learned to add two numbers. The final accuracy on the test set is 100%, indicating that the model perfectly predicted the sum of the test input pairs. This example demonstrates the basic workflow of creating and training a neural network in PyTorch.