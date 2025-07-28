import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch # Only used for initial data loading

def load_and_process_mnist(n_features=4, n_samples=200):
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    idx = (mnist_train.targets == 0) | (mnist_train.targets == 1)
    data = mnist_train.data[idx].float()
    targets = mnist_train.targets[idx].float()
    data = data.reshape(data.shape[0], -1)[:n_samples]
    targets = targets[:n_samples]
    pca = PCA(n_components=n_features)
    data_pca = pca.fit_transform(data.numpy())
    norm = np.max(np.abs(data_pca))
    data_pca /= norm
    targets_np = 2 * targets.numpy() - 1
    return data_pca, targets_np


n_qubits = 4 
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # feature encoding to quantum state using angle embedding
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # standard hidden layer similar to the classical perceptron
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def init_parameters(n_layers, n_qubits):
    return np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits), requires_grad=True)

def accuracy(predictions, labels):
    predicted_labels = np.sign(predictions)
    correct = np.sum(predicted_labels == labels)
    return (correct / len(labels)) * 100

def cost_fn(weights, features, labels):
    predictions = np.array([quantum_circuit(f, weights) for f in features])
    mse = np.mean((predictions - labels) ** 2)
    return mse

def main():
    n_features = 4
    n_layers = 2
    learning_rate = 0.1
    epochs = 20
    batch_size = 10
    n_samples = 400
    features, labels = load_and_process_mnist(n_features=n_features, n_samples=n_samples)

    split_idx = int(0.8 * n_samples)
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    weights = init_parameters(n_layers, n_qubits)
    # back_prop equivalent
    gradient_fn = qml.grad(cost_fn, argnum=0)

    print("\nStarting training...")
    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        # Create batches
        permutation = np.random.permutation(len(train_features))
        train_features_shuffled = train_features[permutation]
        train_labels_shuffled = train_labels[permutation]
        
        total_loss = 0
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features_shuffled[i:i+batch_size]
            batch_labels = train_labels_shuffled[i:i+batch_size]
            gradients = gradient_fn(weights, batch_features, batch_labels)
            weights = weights - learning_rate * gradients
            loss = cost_fn(weights, batch_features, batch_labels)
            total_loss += loss

        avg_loss = total_loss / (len(train_features) / batch_size)
        history["train_loss"].append(avg_loss)
        # valiidation
        val_predictions = np.array([quantum_circuit(f, weights) for f in val_features])
        val_acc = accuracy(val_predictions, val_labels)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1:2d} | Avg Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

    # Plotting
    print("\nTraining complete.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['train_loss'])
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax2.plot(history['val_accuracy'])
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

if __name__ == "__main__":
    main()
