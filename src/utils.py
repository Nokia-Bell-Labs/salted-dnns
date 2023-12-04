import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms


##########
def get_data_loader(args, dataset, split_type="train"):        
    if split_type == "train":                
        _xy = TensorDataset(torch.Tensor(dataset[0][0]), torch.Tensor(dataset[0][1]).long())
        data_loader = DataLoader(_xy, batch_size=args.batch_size, shuffle=True, drop_last=True)            
    elif split_type == "valid":
        _xy = TensorDataset(torch.Tensor(dataset[1][0]), torch.Tensor(dataset[1][1]).long())     
        data_loader = DataLoader(_xy, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    elif split_type == "test":
        _xy = TensorDataset(torch.Tensor(dataset[2][0]), torch.Tensor(dataset[2][1]).long())     
        data_loader = DataLoader(_xy, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    else:
        print("Wrong Split Type!!")
    return data_loader

##########
def get_aug_set(args):
    AUG_SET =   [transforms.RandomHorizontalFlip(p=1.),
                   transforms.RandomErasing(scale=(.02, 0.2), ratio=(.2, 2.2), value=0, p=1.),
                   transforms.RandomPerspective(distortion_scale=0.5, p=1.),               
                   transforms.RandomAdjustSharpness(sharpness_factor=5, p=1.),
                   transforms.RandomAutocontrast(p=1.),
                   ###
                   #transforms.RandomRotation(degrees=(0, 180)),
                   transforms.Compose([
                       transforms.CenterCrop(size=args.image_size*2//3),
                       transforms.Resize(size=args.image_size)]
                   ),
                #   transforms.Compose([
                #       transforms.Pad(padding=10, fill=.0),
                #       transforms.Resize(size=args.image_size)]
                #   ),                      
                   transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
                #   transforms.RandomAffine(degrees=(1, 2), translate=(0.1, 0.3), scale=(0.75, 1.), shear=0.2)
                   ###
                   #, transforms.Lambda(lambda t: t.add(1).div(2)), # From [-1,1] to [0,1]
                   #transforms.Lambda(lambda t: t.mul(2).sub(1)), # From [0,1] to [-1,1]
                ] 
    return AUG_SET

##########
def evaluate(args, model, dataloader):
    correct = 0
    total = 0
    model.to(args.device)
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)        
            
            if args.salt_type== "none":
                new_labels = labels
                outputs = model(inputs)            
            elif args.salt_type== "cnn":
                salts = torch.randint(args.num_classes,(len(labels),))
                salts = salts.to(args.device)
                new_labels = (salts + labels)%args.num_classes        
                salts = torch.reshape(salts, (len(labels),1,1,1)).float()
                outputs = model(inputs, salts)           
            elif args.salt_type== "fc":
                new_labels = torch.zeros_like(labels)
                salts = torch.zeros((len(labels), args.salt_size))
                for j in range(len(labels)):
                    salt_seed = torch.randint(args.num_classes,(1,))[0].item()            
                    new_labels[j] = (salt_seed + labels[j])%args.num_classes
                    means = torch.roll(torch.arange(1., args.num_classes+1), salt_seed+1)
                    means[salt_seed] += 4*args.num_classes
                    means = means/(2*args.num_classes)
                    stds = torch.ones(args.num_classes)/(2*args.num_classes)
                    salts[j] = torch.normal(mean=means, std=stds)                
                salts = salts.to(args.device)
                new_labels = new_labels.to(args.device)
                outputs = model(inputs, salts) 

            _, predicted = torch.max(outputs.data, 1)
            total += new_labels.size(0)
            correct += (predicted == new_labels).sum().item()

    test_acc = 100 * correct / total
    return test_acc

##########
def train_test(args, model, dataset, save_model=False):
    
    if save_model:
        save_path = "results/"+str(args.dataset)+"/"+str(args.exp_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path) 

    loss_func = nn.CrossEntropyLoss()

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.ler_rate, momentum=0.9)
    elif args.optim == "adam": 
        optimizer = optim.Adam(model.parameters(), lr=args.ler_rate)

    best_test_acc = 0
    acc_log = []
    
    for epoch in range(args.epochs):
        
        running_loss = 0.0        
        running_acc = 0
        model.train()
        train_loader = get_data_loader(args, dataset, split_type="train")
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            ###################
            if args.with_aug:
                AUG_SET = get_aug_set(args)
                aug_inputs = torch.stack([AUG_SET[np.random.randint(len(AUG_SET))](img) for img in inputs])
                aug_inputs = torch.clamp(aug_inputs, min=0.0, max=1.0)
                inputs = torch.cat((inputs, aug_inputs), dim=0)
                labels = torch.cat((labels, labels), dim=0)
            ###################
            optimizer.zero_grad()

            if args.salt_type== "none":
                new_labels = labels
                outputs = model(inputs)            
            elif args.salt_type== "cnn":                
                salts = torch.randint(args.num_classes,(len(labels),))
                salts = salts.to(args.device)
                new_labels = (salts + labels)%args.num_classes        
                salts = torch.reshape(salts, (len(labels),1,1,1)).float()
                outputs = model(inputs, salts)            
            elif args.salt_type== "fc":
                new_labels = torch.zeros_like(labels)
                salts = torch.zeros((len(labels), args.salt_size))
                for j in range(len(labels)):
                    salt_seed = torch.randint(args.num_classes,(1,))[0].item()            
                    new_labels[j] = (salt_seed + labels[j])%args.num_classes
                    means = torch.roll(torch.arange(1., args.num_classes+1), salt_seed+1)
                    means[salt_seed] += 4*args.num_classes
                    means = means/(2*args.num_classes)
                    stds = torch.ones(args.num_classes)/(2*args.num_classes)
                    salts[j] = torch.normal(mean=means, std=stds)                
                salts = salts.to(args.device)
                new_labels = new_labels.to(args.device)
                outputs = model(inputs, salts) 

            loss = loss_func(outputs, new_labels)
            loss.backward()
            optimizer.step()
            #
            _, predicted = torch.max(outputs.data, 1)        
            running_acc += ((predicted == new_labels).sum().item())/len(new_labels)
            running_loss += loss.item()
        
        running_loss = running_loss / i 
        running_acc = 100 * running_acc / i

        model.eval()
        test_loader = get_data_loader(args, dataset, split_type="test")
        test_acc = evaluate(args, model, test_loader)
        if test_acc > best_test_acc:                        
            print(f'*******[{epoch + 1}, {i + 1:5d}] train loss: {running_loss:.3f} \| train acc: {running_acc:.3f} \| test acc: {test_acc:.3f}')
            best_test_acc = test_acc
            if save_model:
                torch.save(model.state_dict(), save_path+"/best_model.pt")
        else:
            print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss:.3f} \| train acc: {running_acc:.3f} \| test acc: {test_acc:.3f}')

        running_loss = 0.0
        acc_log.append([running_acc, test_acc])

    acc_log = np.array(acc_log)
    if save_model:
        np.save(save_path+"/accuracy_log.npy", acc_log)
    
    print('Finished Training')
    
    model.load_state_dict(torch.load(save_path+"/best_model.pt", 
                           map_location=torch.device(args.device)))
    test_loader = get_data_loader(args, dataset, split_type="test")
    best_test_acc = evaluate(args, model, test_loader)
    print(f'Best Test Accuracy: {best_test_acc} %')
