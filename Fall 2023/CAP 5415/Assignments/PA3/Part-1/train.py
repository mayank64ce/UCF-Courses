import torch
import torch.nn as nn
from argparse import ArgumentParser
from torchvision import transforms, datasets
from dataloader import get_mnist_dataloaders
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import wandb



def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''

    # Set model to train mode before each epoch
    model.train()

    # Empty list to store losses
    losses = []

    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, _ = batch_sample

        # Push data/label to correct device
        data = data.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data)

        # ======================================================================
        # Compute loss based on criterion
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign correct loss function.
        loss = criterion(output, data)

        # Computes gradient based on final loss
        loss.backward()

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()

    train_loss = float(np.mean(losses))
    print('Train set: Average loss: {:.4f}\n'.format(
        train_loss))
    return train_loss

def test(model, device, test_loader, optimizer, criterion, epoch, batch_size):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''

    # Set model to eval mode to notify all layers.
    model.eval()

    losses = []

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, _ = sample
            data = data.to(device)

            # Predict for data by doing forward pass
            output = model(data)

            # ======================================================================
            # Compute loss based on same criterion as training
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign correct loss function.
            # Compute loss based on same criterion as training
            loss = criterion(output, data)

            # Append loss to overall test loss
            losses.append(loss.item())

    test_loss = float(np.mean(losses))

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))

    return test_loss

def main(args):
    wandb.login(key="Your API key here")
    run = wandb.init(
    # Set the project where this run will be logged
    project=f"PA3 Model training in {args.mode} mode.",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "mode" : args.mode
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch device selected: ", device)
    print(
        f"Mode: {args.mode}, Learning Rate: {args.learning_rate}, Epochs: {args.epochs}")
    
    if args.mode == "FC":
        from models import AutoencoderFC
        model = AutoencoderFC().to(device)
    elif args.mode == "CNN":
        from models import AutoencoderCNN
        model = AutoencoderCNN().to(device)
    
    model_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"Total number of trainable parameters in model = {model_params}.")
    print(f"Total number of trainable parameters in encoder = {encoder_params}.")
    print(f"Total number of trainable parameters in decoder = {decoder_params}.")
    
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.learning_rate)
    
    train_dataloader, test_dataloader = get_mnist_dataloaders(args)

    best_loss = math.inf
    best_epoch = 0.0
    loss_train = []
    loss_test = []

    for epoch in range(1, args.epochs + 1):
        print(f"================= Running Epoch : {epoch} =================")
        train_loss = train(model, device, train_dataloader,
                                           optimizer, criterion, epoch, args.batch_size)
        test_loss = test(
            model, device, test_dataloader, optimizer, criterion, epoch, args.batch_size)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
        
        loss_train.append(train_loss)
        loss_test.append(test_loss)

        wandb.log({"train_loss": train_loss, "test_loss": test_loss})
    
    loss_train = np.array(loss_train)
    loss_test = np.array(loss_test)

    fig = plt.figure(1)
    epochs = range(1, args.epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_test, 'b', label='Test loss')
    plt.title('Model mode: '+str(args.mode)+' Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig.savefig('plots/loss_plot mode: '+str(args.mode)+'.png')

    print(f"The model converged at epoch number : {best_epoch}")

    # save model

    save_folder = "saved_weights"

    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, args.model_save_path)
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights can be found in {save_path} .")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", help="Learning rate", default=1e-3, type=float)
    parser.add_argument("--mode", help="Autoencoder Mode", default="CNN", choices=["FC", "CNN"])
    parser.add_argument("--epochs", help="Number of Training epochs", default=10, type=int)
    parser.add_argument("--exp_name", help="Name of experiment")
    parser.add_argument("--batch_size", help="Batch size for training", default=10, type=int)
    parser.add_argument("--model_save_path", help="Path for saving model weights")
    args = parser.parse_args()
    main(args)
