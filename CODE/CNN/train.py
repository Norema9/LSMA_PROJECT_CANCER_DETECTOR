import os
import torch
import torch.optim as optim
import torch.nn as nn
from cnn import CNN_CANCER_DETECTOR
from dataset import get_data_loader
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, log_dir, checkpoint_step, start_epoch=0, best_accuracy=0.0):
    """
    Trains the neural network model and validates it after each epoch.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for the training process.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the training on (CPU or GPU).
        checkpoint_dir (str): Directory to save the model checkpoints.
        log_dir (str): Directory to save the TensorBoard logs.
        checkpoint_step (int): Number of steps after which to save a checkpoint.
        start_epoch (int): Epoch number to start training from, for resuming training.
        best_accuracy (float): Best validation accuracy achieved, for saving the best model.

    """
    writer = SummaryWriter(log_dir=log_dir)  # Initialize TensorBoard writer

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        print(f"\nEpoch : {epoch}")
        for i, (images, labels, _) in tqdm(enumerate(train_loader)):  # Iterate over the training data
            images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute the loss
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the model parameters

            l = loss.item()  # Get the loss value
            running_loss += l
            
            writer.add_scalar('Loss/Train', l, epoch)  # Log the training loss

            # Save checkpoint
            if (i + 1) % checkpoint_step == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"latest_checkpoint.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'best_accuracy': best_accuracy
                }, checkpoint_path)  # Save the model checkpoint
        
        avg_train_loss = running_loss / len(train_loader)  # Compute average training loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():  # Disable gradient calculation for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())  # Store true labels
                all_preds.extend(predicted.cpu().numpy())  # Store predicted labels
        
        avg_val_loss = val_loss / len(val_loader)  # Compute average validation loss
        accuracy = 100 * correct / total  # Compute validation accuracy
        precision = precision_score(all_labels, all_preds, average='weighted')  # Compute precision
        recall = recall_score(all_labels, all_preds, average='weighted')  # Compute recall
        f1 = f1_score(all_labels, all_preds, average='weighted')  # Compute F1-score
        
        print(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%")
        print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)  # Log validation loss
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)  # Log validation accuracy
        writer.add_scalar('Precision/Validation', precision, epoch)  # Log validation precision
        writer.add_scalar('Recall/Validation', recall, epoch)  # Log validation recall
        writer.add_scalar('F1-Score/Validation', f1, epoch)  # Log validation F1-score
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)  # Save the best model
            print(f"Best model saved with accuracy: {accuracy}%")

        # Save classification report
        report = classification_report(all_labels, all_preds, target_names=['notumor', 'meningioma', 'pituitary'], output_dict=True)
        with open(os.path.join(checkpoint_dir, f'classification_report_epoch_{epoch+1}.txt'), 'w') as f:
            f.write(f"Epoch {epoch+1}\n")
            f.write(str(report))  # Save classification report to file
    
    writer.close()  # Close the TensorBoard writer
    print("Finished Training")

def main():
    """
    Main function to set up directories, load data, initialize model, and start training.
    """
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT")  # Change to the project directory
    sys.path.append("CODE\CNN")  # Add CNN code to the system path
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'  # Disable oneDNN optimizations

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model, criterion, and optimizer
    model = CNN_CANCER_DETECTOR(channel_size=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Directory of the training and testing images
    train_image_dir = r"DATA\processed\train"
    test_image_dir = r"DATA\processed\test"

    # Load data
    train_loader = get_data_loader(image_dir=train_image_dir, batch_size=32, shuffle=True)
    val_loader = get_data_loader(image_dir=test_image_dir, batch_size=32, shuffle=False)
    
    # Create checkpoint directory
    checkpoint_dir = r"LOGS_CNN\checkpoints"
    log_dir = r"LOGS_CNN\log_dir"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Training parameters
    num_epochs = 30
    checkpoint_step = 100  # Save checkpoint every 100 steps
    
    # Check for existing checkpoint
    start_epoch = 0
    best_accuracy = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        print(f"Resuming training from epoch {start_epoch} with best accuracy {best_accuracy}%")

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, log_dir, checkpoint_step, start_epoch, best_accuracy)

if __name__ == "__main__":
    main()
