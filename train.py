import torch
from torch import nn
from data_processing.data_preprocessing import *
from model.model import *
import os

check_interval = 1
num_epochs = 10
autoencoder = Autoencoder()

optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
loss = nn.MSELoss()

autoencoder.train()

test_loss_avg = []

print('Training ...')

if os.path.exists('model_checkpoint.pt'):
    checkpoint = torch.load('model_checkpoint.pt')
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = checkpoint['epoch']
    print("Model incarcat din checkpoint")

for epoch in range(num_epochs):
    test_loss_avg.append(0)
    num_batches = 0
    print('--------')
        
    for test_noisy_batch, _ in trainnoisyloader:

        test2, _ = next(iter(trainnoisyloader))
        test_batch, _ = next(iter(trainloader))
        # print(test_batch.size())

        if test_batch is None:
            break  

        image_batch_recon = autoencoder(test2)
        print(image_batch_recon.size())
        
        reconstruction_loss = loss(image_batch_recon, test_batch)

        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()

        if epoch % check_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_checkpoint.pt')


        test_loss_avg[-1] += reconstruction_loss.item()
        num_batches += 1

    test_loss_avg[-1] /= num_batches
    print('Epoca [%d / %d] are Average Reconstruction Loss: %f' % (epoch+1, num_epochs, test_loss_avg[-1]))


