import torch
import torchvision

import dataset.transforms as T

from pathlib import Path


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    A simplified torchvision.datasets.CocoDetection class (keeps
    all the features needed for object detection only).
    """

    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.transforms = transforms
        self.convert_func = ConvertCocoAnnotations()

    def __getitem__(self, idx):  # TODO: refactor
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}  # TODO: really needed?
        img, target = self.convert_func(img, target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


class ConvertCocoAnnotations():
    """
    A callable class for converting coco annotations to a format
    more suitable for PyTorch.
    """

    def __call__(self, image, target):
        """
        Converts the target from the list of dictionaries (one for
        each object) to a single dictionary with tensors as values (more suitable
        for PyTorch).

        Does some bboxes sanity checks by the way.
        """

        w, h = image.size

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        anno = target['annotations']

        # TODO: descr
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # Convert xywh to xyxy

        # Handle boxes crossing img boundaries
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Leave correct boxes only
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # Some more additional info
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])

        # Build the new-style dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def get_transforms(subset):
    """
	Returns custom transforms (analogous to PyTorch ones) working
    both for images and labels.
	"""

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if subset == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if subset == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])


def build_dataset(coco_dir: str, subset: str):
    assert subset in ('train', 'val'), f'Unknown subset type: {subset}'

    root = Path(coco_dir) / f'{subset}2017'
    ann_file = Path(coco_dir) / 'annotations' / f'instances_{subset}2017.json'
    transforms = get_transforms(subset)
    
    dataset = CocoDetection(str(root), str(ann_file), transforms=transforms)
    
    return dataset
