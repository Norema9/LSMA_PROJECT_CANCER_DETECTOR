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
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch : {epoch}")
        for i, (images, labels, _) in tqdm(enumerate(train_loader)):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss.item()
            running_loss += l
            
            writer.add_scalar('Loss/Train', l, epoch)

            # Save checkpoint
            if (i + 1) % checkpoint_step == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"latest_checkpoint.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'best_accuracy': best_accuracy
                }, checkpoint_path)
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%")
        print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        writer.add_scalar('Precision/Validation', precision, epoch)
        writer.add_scalar('Recall/Validation', recall, epoch)
        writer.add_scalar('F1-Score/Validation', f1, epoch)
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {accuracy}%")

        # Save classification report
        report = classification_report(all_labels, all_preds, target_names=['notumor', 'meningioma', 'pituitary'], output_dict = True)
        with open(os.path.join(checkpoint_dir, f'classification_report_epoch_{epoch+1}.txt'), 'w') as f:
            f.write(f"Epoch {epoch+1}\n")
            f.write(str(report))
    
    writer.close()
    print("Finished Training")

def main():
    os.chdir(r"C:\Users\maron\OneDrive\Bureau\PROJECT") # Adapt this to the location of the PROJECT directory
    sys.path.append("CODE\CNN")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model, criterion, and optimizer
    model = CNN_CANCER_DETECTOR(channel_size=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # directory of the training and testing images
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
