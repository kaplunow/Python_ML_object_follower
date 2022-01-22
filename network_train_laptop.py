import torch
import torchvision

# data import
dataset = torchvision.datasets.ImageFolder(
    'dataset',  # folder name
    torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

# split into two sets
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 10, 10])

# create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

# get our model of choice, pretrained
model = torchvision.models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)  # replace last stage with custom one
device = torch.device('cuda')
model = model.to(device)  # put model into GPU


NUM_EPOCHS = 5
BEST_MODEL_PATH = 'network_ball_model.pth'  # network file name
best_accuracy = 0.0

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):

    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)  # move train data to GPU
        optimizer.zero_grad()  # clean gradients
        outputs = model(images)  # get models response
        loss = torch.nn.functional.cross_entropy(outputs, labels)  # calculate loss
        loss.backward()  # calculate gradients
        optimizer.step()  # optimize models parameters

    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f' % (epoch, test_accuracy))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy
