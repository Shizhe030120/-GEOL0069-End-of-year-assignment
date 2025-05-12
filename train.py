import os
import sys
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # Training and testing datasets are stored in data_set
    image_path = os.path.join('/root/autodl-tmp/1200120012001200/', "data_set")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = AlexNet(num_classes=10, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.00005)

    epochs = 20
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)

    # For storing loss and accuracy of each epoch
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0  # training accuracy
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            # Calculate training accuracy
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # Record training loss and accuracy
        train_losses.append(running_loss / train_steps)
        train_accuracies.append(train_acc / train_num)

        # validate
        net.eval()
        val_loss = 0.0  # validation loss
        acc = 0.0  # accumulate accurate number / epoch
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # Calculate validation loss
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                # Record predictions and true labels
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        val_accurate = acc / val_num
        val_accuracies.append(val_accurate)
        val_losses.append(val_loss / len(validate_loader))

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_losses[-1], train_accuracies[-1], val_losses[-1], val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    # Output classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, digits=2)
    print(report)

    # Draw classification report as table
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 0.01, report, {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.title("Classification Report")
    plt.savefig("classification_report.png")  # Save classification report image
    plt.show()

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Draw confusion matrix heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")  # Save confusion matrix image
    plt.show()

    print('Finished Training')

    # Set x-axis interval to 1
    x_ticks = range(1, epochs + 1)
    
    # Draw loss curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xticks(x_ticks)  # Set x-axis with interval of 1
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.grid(True)
    plt.legend()

    # Draw accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xticks(x_ticks)  # Set x-axis with interval of 1
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics_curve.png")  # Save loss and accuracy curve image
    plt.show()


if __name__ == '__main__':
    main()