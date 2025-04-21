import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import os

class LabelRemappingDataset(Dataset):
    def __init__(self, subset, original_class_to_idx, target_classes_map, transform=None):
        self.subset = subset
        self._transform = transform
        self.original_idx_to_target_idx = {
            original_idx: target_classes_map[class_name]
            for class_name, original_idx in original_class_to_idx.items()
            if class_name in target_classes_map
        }
        print(f"Label remapping created: {self.original_idx_to_target_idx}")
        if len(self.original_idx_to_target_idx) != len(target_classes_map):
             print("Warning: Not all target classes were found in the original dataset's mapping.")


    def __getitem__(self, index):
        img, original_label_idx = self.subset[index]

        if self.transform:
            img = self.transform(img)

        target_label = self.original_idx_to_target_idx.get(original_label_idx, -1)
        if target_label == -1:
            print(f"Error: Could not map original label index {original_label_idx} at subset index {index}.")
        return img, target_label

    def __len__(self):
        return len(self.subset)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224, train_split_ratio: float = 0.8):
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found or is not a directory.")
        return None, None

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    except FileNotFoundError:
        print(f"Error: Could not find image data in '{data_dir}'. Ensure it contains subdirectories for classes.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return None, None

    if not full_dataset.classes or len(full_dataset) == 0:
        print(f"Error: No classes or images found in '{data_dir}'. Check the directory structure.")
        return None, None

    print(f"Dataset loaded successfully. Found {len(full_dataset)} images in {len(full_dataset.classes)} classes: {full_dataset.classes}")

    target_classes = {'no': 0, 'yes': 1}
    target_class_names = list(target_classes.keys())

    valid_indices = []
    original_labels_for_subset = []
    try:
        print(f"Initial dataset classes: {full_dataset.classes}")
        print(f"Initial class_to_idx: {full_dataset.class_to_idx}")
        for i, (path, class_idx) in enumerate(full_dataset.samples):
            class_name = full_dataset.classes[class_idx]
            if class_name in target_classes:
                valid_indices.append(i)
                original_labels_for_subset.append(class_idx)
    except Exception as e:
         print(f"Error processing dataset samples: {e}")
         return None, None

    if not valid_indices:
        print(f"Error: No samples found for target classes '{', '.join(target_class_names)}' in '{data_dir}'.")
        return None, None

    filtered_subset = Subset(full_dataset, valid_indices)
    final_dataset = LabelRemappingDataset(filtered_subset, full_dataset.class_to_idx, target_classes, transform=None)

    total_final_size = len(final_dataset)
    train_size = int(train_split_ratio * total_final_size)
    val_size = total_final_size - train_size

    if total_final_size < 2:
        print("Error: Final dataset needs at least 2 samples for a train/validation split.")
        return None, None
    if train_size == 0:
        train_size = 1
        val_size = total_final_size - 1
    elif val_size == 0:
        val_size = 1
        train_size = total_final_size - 1

    print(f"Splitting final dataset: {train_size} training samples, {val_size} validation samples.")

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(final_dataset, [train_size, val_size], generator=generator)

    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms
    print("Applied specific transforms to train and validation subsets.")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"DataLoaders created successfully with remapped labels ({target_classes}) and specific transforms.")

    return train_loader, val_loader

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_directory = os.path.join(project_root, 'archive')

    print(f"Looking for data in: {data_directory}")

    train_dl, val_dl = get_dataloaders(data_dir=data_directory)

    if train_dl and val_dl:
        print("\nTesting DataLoaders...")
        try:
            train_features, train_labels = next(iter(train_dl))
            print(f"\nTraining Batch Shape:")
            print(f"Features batch shape: {train_features.size()}")
            print(f"Labels batch shape: {train_labels.size()}")
            print(f"Sample label: {train_labels[0]}")
            original_class_to_idx = train_dl.dataset.dataset.subset.dataset.class_to_idx
            print(f"Original class mapping: {original_class_to_idx}")
            print(f"Remapped label mapping used: {train_dl.dataset.dataset.original_idx_to_target_idx}")

            val_features, val_labels = next(iter(val_dl))
            print(f"\nValidation Batch Shape:")
            print(f"Features batch shape: {val_features.size()}")
            print(f"Labels batch shape: {val_labels.size()}")
            print(f"Sample label: {val_labels[0]}")

            print("\nData loader test completed.")
        except StopIteration:
            print("\nCould not retrieve a batch. Check if the dataset split resulted in empty sets or if batch_size is larger than dataset size.")
        except Exception as e:
            print(f"\nAn error occurred during DataLoader testing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nDataLoaders could not be created.")