import torch

def test_boolean(network, loss, dataloader, dev):
    network.eval()
    total = 0
    correct1 = 0
    total_loss = 0
    with torch.no_grad():
        for idx, (data, target, task) in enumerate(dataloader):
            data = data.to(dev)
            target = target.to(dev)
            output = network(data.float(), task, train=False)
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

def train_network(network, loss, optimizer, train_loader, train_loader2, validation_loader, test_loader, dev, epochs, scheduler):

    train_loss = []
    train_accuracy = []
    
    validation_loss = []
    validation_accuracy = []

    test_accuracy = []
    test_loss = []

    for epoch in range(epochs):
        network.train()
        for batch_idx, (data, target, task) in enumerate(train_loader):
            data = data.to(dev)
            target = target.to(dev)
            optimizer.zero_grad()
            output = network(data.float(), task)
            batch_loss = loss(output, target.float())
            batch_loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.item()))

        train_avg_loss, train_acc1 = test_boolean(network, loss, train_loader2, dev)
        train_loss.append(train_avg_loss)
        train_accuracy.append(train_acc1)

        val_avg_loss, val_acc1 = test_boolean(network, loss, validation_loader, dev)
        validation_loss.append(val_avg_loss)
        validation_accuracy.append(val_acc1)

        avg_loss, acc1 = test_boolean(network, loss, test_loader, dev)
        test_accuracy.append(acc1)
        test_loss.append(avg_loss)

        scheduler.step()

    return network, validation_loss, train_loss, test_loss, validation_accuracy, train_accuracy, test_accuracy