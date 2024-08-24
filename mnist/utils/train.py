import torch

def test_boolean(network, loss, dataloader, dev):
    network.eval()
    total = 0
    correct1 = 0
    total_loss = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(dev)
            target = target.to(dev)
            output = network(data.float(), train=False)
            output = output
            total_loss += loss(output,target.float())
            inds = output > 0.5
            output = inds.float()
            total += torch.numel(target)
            correct1 += torch.sum(output==target)

    acc = 100.0 * correct1 / total
    total_loss = total_loss / (idx+1)

    print('Top 1 Accuracy =', acc)
    print('Average Loss =', total_loss)

    return total_loss.detach().cpu().numpy(), acc.detach().cpu().numpy()

def train_network(network, loss, optimizer, train_loader, val_loader, test_loader1, test_loader2, dev, epochs, scheduler):
    
    validation_accuracy = []
    validation_loss = []

    test_accuracy1 = []
    test_loss1 = []

    test_accuracy2 = []
    test_loss2 = []

    for epoch in range(epochs):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            target = target.to(dev)
            optimizer.zero_grad()
            output = network(data.float())
            batch_loss = loss(output, target.float())
            batch_loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.item()))
        
        val_loss, val_acc = test_boolean(network, loss, val_loader, dev)
        validation_accuracy.append(val_acc)
        validation_loss.append(val_loss)

        avg_loss1, acc1 = test_boolean(network, loss, test_loader1, dev)
        test_accuracy1.append(acc1)
        test_loss1.append(avg_loss1)

        avg_loss2, acc2 = test_boolean(network, loss, test_loader2, dev)
        test_accuracy2.append(acc2)
        test_loss2.append(avg_loss2)

        scheduler.step()

    return network, validation_loss, test_loss1, test_loss2, validation_accuracy, test_accuracy1, test_accuracy2