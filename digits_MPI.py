#!/usr/bin/env python
import sys
from mpi4py import MPI   
import torch
from torchvision import transforms
import torchvision.datasets as datasets

import sys
sys.path.insert(0, 'ArchTests/')


size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.COMM_WORLD.Get_name()

comm = MPI.COMM_WORLD

if (rank == 0):
    from model0 import Model
if (rank == 1):
    from model1 import Model
if (rank == 2):
    from model2 import Model
if (rank == 3):
    from model3 import Model


    
    
## Train the model
    
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

mnist_valset, mnist_testset = torch.utils.data.random_split(mnist_testset, [int(0.9 * len(mnist_testset)), int(0.1 * len(mnist_testset))])

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)

if (rank == 0):
    print("Training dataset size: " + str( len(mnist_trainset)))
    print("Validation dataset size: " + str( len(mnist_valset)))
    print("Testing dataset size: " + str( len(mnist_testset)))
    print("Commencing training networks using MPI4PY")


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


no_epochs = 1
train_loss = list()
val_loss = list()
best_val_loss = 1
for epoch in range(no_epochs):
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    # training
    for itr, (image, label) in enumerate(train_dataloader):

        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    total_train_loss = total_train_loss / (itr + 1)
    train_loss.append(total_train_loss)

    # validation
    model.eval()
    total = 0
    for itr, (image, label) in enumerate(val_dataloader):

        pred = model(image)

        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy = float(total) / len(mnist_valset)

    total_val_loss = total_val_loss / (itr + 1)
    val_loss.append(total_val_loss)

    print('\nFrom rank {}, Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(rank, epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, total_val_loss))
        torch.save(model.state_dict(), "TrainedModels/model" + str(rank) + ".dth")


# test model
model.load_state_dict(torch.load("TrainedModels/model" + str(rank) + ".dth"))
model.eval()

results = list()
correct = 0
for itr, (image, label) in enumerate(test_dataloader):

    pred = model(image)
    pred = torch.nn.functional.softmax(pred, dim=1)

    for i, p in enumerate(pred):
        if label[i] == torch.max(p.data, 0)[1]:
            correct = correct + 1
            results.append((image, torch.max(p.data, 0)[1]))

test_accuracy = float(correct)/len(mnist_testset)
print('Test accuracy ' + str(test_accuracy))


#communcate among the processors and find the minimum loss which corresponds to the Neural network which trained the best in the first epoch
max_class_acc = comm.allreduce(test_accuracy, op=MPI.MAX)

if (max_class_acc == test_accuracy): 
    print("\nRank " + str(rank) + " trained the best network(in terms of classification accuracy on the test data)! The architecture can be found in ArchTests directory under name model" + str(rank) + " and is saved in the TrainedModels directory under name model" + str(rank) + ".dth. You can continue training with that model or just train from scratch.")
