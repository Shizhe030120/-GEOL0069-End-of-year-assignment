import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from model import resnet50

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Image path
    img_path = "./1.png"  # Replace with your image path
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()
    # Transform image
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0).to(device)
    # Read class indices
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # Create model
    model = resnet50(num_classes=5).to(device)
    model.load_state_dict(torch.load("./myresNet50.pth", map_location=device))
    # Define a function to get feature maps
    def get_activation_maps(model, input, output):
        activation_maps = output.cpu().detach()
        show_feature_maps(activation_maps)
    # Assume the name of the last convolutional layer is layer4[2].conv3
    last_conv_layer = model.layer4[2].conv3
    handle = last_conv_layer.register_forward_hook(get_activation_maps)
    # Model evaluation mode
    model.eval()
    # Prediction
    with torch.no_grad():
        output = model(img)
        output = output.to('cpu')
        predict = torch.softmax(output, dim=1)
        predict_cla = torch.argmax(predict).cpu().numpy()
    print_res = "class: {}   prob: {:.3f}".format(class_indict[str(predict_cla)], predict[0, predict_cla].item())
    print(print_res)
    # Remove hook
    handle.remove()
    # Convert sparse tensor to dense tensor (if needed)
    if img.is_sparse:
        img = img.to_dense()
    # Now img is a dense tensor, remove batch dimension
    img = img.squeeze(0)  # Convert to [C, H, W]
    # Denormalize to display image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    img = img * std + mean
    img = img.clamp(0, 1)  # Ensure values are in range [0, 1]
    img = img.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C]
    plt.imshow(img)
    plt.title(print_res)
    plt.show()

def show_feature_maps(activation_maps):
    activation_maps = activation_maps[0]
    n_features = activation_maps.size(0)
    nrows = int(np.ceil(np.sqrt(n_features)))
    ncols = int(np.ceil(n_features / nrows))
    plt.figure(figsize=(15, 15))
    for i in range(n_features):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(activation_maps[i].cpu(), cmap='viridis')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()