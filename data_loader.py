from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size):
    # Tranform to tensor
    transform = transforms.ToTensor()

    # Load dataset 
    data_dir = "assessment-data"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
