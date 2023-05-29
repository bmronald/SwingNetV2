from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import colorama
from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

if __name__ == '__main__':

    # Calculate and update the evaluation metrics
    def update_metrics(logits, labels, predicted_labels):
        # Calculate loss
        loss = criterion(logits, labels)
        # Forward pass and prediction
        predicted = torch.argmax(logits, dim=1)
        # Append predicted labels to the list
        predicted_labels += predicted.cpu().numpy().tolist()  
        accuracy = accuracy_score(labels.cpu(), predicted.cpu())
        precision = precision_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
        f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
        # Update metrics
        losses.update(loss.item(), logits.size(0))
        accuracies.update(accuracy, logits.size(0))
        precisions.update(precision, logits.size(0))
        recalls.update(recall, logits.size(0))
        f1_scores.update(f1, logits.size(0))
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Append predicted labels to the list
        predicted_labels += predicted.cpu().numpy().tolist()  

    # Initialize colorama
        colorama.init()
    # training configuration
    split = 1
    iterations = 2000
    it_save = 100  # save model every 100 iterations
    n_cpu = 6
    seq_length = 64
    bs = 8  # batch size
    k = 10  # frozen layers

    # AverageMeter instances to keep track of the accuracy, precision, recall, F1 score, and losses
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1_scores = AverageMeter()
    losses = AverageMeter()


    # create the model and move it to the device
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    model.cuda()

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # create models directory (if necessary)
    if not os.path.exists('models'):
        os.mkdir('models')
    
    # training loop 
    predicted_labels = []
    i = 0
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)
            labels = labels.view(bs*seq_length)
            # update metrics
            update_metrics(logits, labels, predicted_labels)
            # console output 
            print(colorama.Fore.GREEN + f'Iteration: {i}\tLoss: {losses.val:.4f} (Avg: {losses.avg:.4f})\tAccuracy: {accuracies.avg:.4f}'
                f'\tPrecision: {precisions.avg:.4f}\tRecall: {recalls.avg:.4f}\tF1 Score: {f1_scores.avg:.4f}'
                + colorama.Style.RESET_ALL)
            i += 1
            # save model 
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))

            if i == iterations:
                break
           
    # reset colorama
    colorama.deinit()




