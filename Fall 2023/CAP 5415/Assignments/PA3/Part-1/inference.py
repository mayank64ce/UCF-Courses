import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def get_figure_for_digit(digit, selected_images, model, device, args):
    digits = []
    for i in range(2):
        digits.append(selected_images[digit][i][None,None,:,:])
    
    digit_batch = torch.cat(digits, dim=0)

    with torch.no_grad():
        digit_batch = digit_batch.to(device).float()
        output = model(digit_batch)
        output = output.cpu()
        digit_batch = digit_batch.cpu()
        plt.figure(figsize=(8, 2))
        for i in range(2):
            plt.subplot(1, 4, 2*i + 1)
            plt.imshow(digit_batch[i].squeeze(), cmap="gray")
            plt.title("Input")
            plt.axis('off')
            plt.subplot(1, 4, 2*i + 2)
            plt.imshow(output[i].squeeze(), cmap="gray")
            plt.title("Output")
            plt.axis('off')
        plt.savefig(f"figures/{args.mode}/{digit}.png")
        plt.show()
    print("Done with digit: ", digit)
    

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_dataset = datasets.MNIST('./data/', train=False,
                              transform=transform)
    if args.mode == "FC":
        from models import AutoencoderFC
        model = AutoencoderFC().to(device)
        model.load_state_dict(torch.load("./saved_weights/model1.pth"))
        model.eval()
    else:
        from models import AutoencoderCNN
        model = AutoencoderCNN().to(device)
        model.load_state_dict(torch.load("./saved_weights/model2.pth"))
        model.eval()
    num_images = 2
    selected_images = {}

    for digit in range(10):
        indices = torch.where(test_dataset.targets == digit)[0][:num_images]
        selected_images[digit] = test_dataset.data[indices]
    
    for digit in range(10):
        get_figure_for_digit(digit, selected_images, model, device, args)
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", help="Autoencoder Mode", default="CNN", choices=["FC", "CNN"])
    args = parser.parse_args()
    main(args)