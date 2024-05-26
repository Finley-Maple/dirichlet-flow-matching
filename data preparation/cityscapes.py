import torch
import os.path
import transforms as trf
import torchvision.transforms as tv_transforms
from torch.utils.data import TensorDataset, DataLoader

class TransformedTensorDataset(TensorDataset):
	"""
	Wrapper for TensorDataset with transforms pipeline
	"""
	def __init__(self, data_tensor, transforms=None):
		super(TransformedTensorDataset, self).__init__(data_tensor)
		self.transforms = transforms
	
	def __getitem__(self, idx):
		(x,) = super(TransformedTensorDataset, self).__getitem__(idx)
		if self.transforms is None:
			return x
		return self.transforms(x)

class CityscapesSegmentations(object):
	"""
	Segmentations of cityscapes images, reduced to 8 class categories
	and with spatial downscaling
	"""
	def __init__(self, dataset_params):
		super(CityscapesSegmentations, self).__init__()
		assert dataset_params["num_classes"] == 8
		self.num_classes = dataset_params["num_classes"]
		self.scale = dataset_params["scale"]
		self.spatial_dims = (int(1024*self.scale), int(2048*self.scale))
		self.tensor_data = None
		self.smoothing = dataset_params["integer_smoothing"]

	def load_data(self, split="train"):
		data_fpath = os.path.join(f"data/image/cityscapes/cityscapes_{split}_{self.scale}.pt")
		assert os.path.isfile(data_fpath), f"No cityscapes data at {data_fpath}"
		print("Loading cityscapes segmentations from", data_fpath)
		self.tensor_data = torch.load(data_fpath)
		self.transforms = tv_transforms.Compose([
			trf.LabelingToAssignment(self.num_classes),
			trf.SmoothSimplexCorners(self.smoothing)
		])
		self.dataset = TransformedTensorDataset(self.tensor_data, self.transforms)

	def tensor_format(self):
		return (-1, self.num_classes, *self.spatial_dims)

	def dataloader(self, split="train", **kwargs):
		if self.tensor_data is None:
			self.load_data(split)
		return DataLoader(self.dataset, shuffle=(split == "train"), **kwargs)

if __name__ == "__main__":
    # Define dataset parameters
    dataset_params = {
        "num_classes": 8,
        "scale": 0.5,
        "integer_smoothing": 1
    }

    # Initialize datasets and dataloaders
    cityscapes = CityscapesSegmentations(dataset_params)
    train_loader = cityscapes.dataloader(split="train", batch_size=8, num_workers=4)
    val_loader = cityscapes.dataloader(split="val", batch_size=8, num_workers=4)

    # Example of using the dataloaders
    for images in train_loader:
        print(images.shape)
        break  # Just print one batch to check
      
# class TransformedTensorDataset(TensorDataset):
#     """
#     Wrapper for TensorDataset with transforms pipeline
#     """
#     def __init__(self, image_tensor, mask_tensor, transforms=None):
#         super(TransformedTensorDataset, self).__init__()
#         self.image_tensor = image_tensor
#         self.mask_tensor = mask_tensor
#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.image_tensor)

#     def __getitem__(self, idx):
#         image = self.image_tensor[idx]
#         mask = self.mask_tensor[idx]
#         if self.transforms:
#             image, mask = self.transforms(image, mask)
#         return image, mask

# class CityscapesSegmentations(object):
#     """
#     Segmentations of cityscapes images, reduced to 8 class categories
#     and with spatial downscaling
#     """
#     def __init__(self, dataset_params):
#         super(CityscapesSegmentations, self).__init__()
#         assert dataset_params["num_classes"] == 8
#         self.num_classes = dataset_params["num_classes"]
#         self.scale = dataset_params["scale"]
#         self.spatial_dims = (int(1024*self.scale), int(2048*self.scale))
#         self.image_tensor = None
#         self.mask_tensor = None
#         self.smoothing = dataset_params["integer_smoothing"]

#     def load_data(self, split="train"):
#         image_fpath = os.path.join(f"data/image/cityscapes/cityscapes_{split}_{self.scale}_images.pt")
#         mask_fpath = os.path.join(f"data/image/cityscapes/cityscapes_{split}_{self.scale}_masks.pt")
#         assert os.path.isfile(image_fpath), f"No cityscapes image data at {image_fpath}"
#         assert os.path.isfile(mask_fpath), f"No cityscapes mask data at {mask_fpath}"
#         print("Loading cityscapes images from", image_fpath)
#         print("Loading cityscapes masks from", mask_fpath)
#         self.image_tensor = torch.load(image_fpath)
#         self.mask_tensor = torch.load(mask_fpath)
#         self.transforms = tv_transforms.Compose([
#             trf.LabelingToAssignment(self.num_classes),
#             trf.SmoothSimplexCorners(self.smoothing),
#             trf.ToTensor(),
#             trf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ])
#         self.dataset = TransformedTensorDataset(self.image_tensor, self.mask_tensor, self.transforms)

#     def tensor_format(self):
#         return (-1, self.num_classes, *self.spatial_dims)

#     def dataloader(self, split="train", **kwargs):
#         if self.image_tensor is None or self.mask_tensor is None:
#             self.load_data(split)
#         return DataLoader(self.dataset, shuffle=(split == "train"), **kwargs)
