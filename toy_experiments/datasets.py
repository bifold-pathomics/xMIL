import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm


def get_number_mil_dataset(
    dataset_type,
    num_numbers,
    num_bags,
    num_instances,
    features_type="mnist_resnet18",
    sampling="hierarchical",
    noise=1.0,
    threshold=1,
    features_path=None,
):
    if dataset_type == "smil":
        dataset_cls = SMILDataset
    elif dataset_type == "four_bags":
        dataset_cls = FourBagsDataset
    elif dataset_type == "pos_neg":
        dataset_cls = PosNegDataset
    elif dataset_type == "adjacent_pairs":
        dataset_cls = AdjacentPairsDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    dataset = dataset_cls(
        num_numbers=num_numbers,
        num_bags=num_bags,
        num_instances=num_instances,
        features_type=features_type,
        sampling=sampling,
        noise=noise,
        threshold=threshold,
        features_path=features_path,
    )
    return dataset


def get_MNIST_features(root, download=True):
    transform_1 = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    # Load dataset
    trainset = torchvision.datasets.MNIST(
        root=root, train=True, download=download, transform=transform_1
    )
    testset = torchvision.datasets.MNIST(
        root=root, train=False, download=download, transform=transform_1
    )
    train_loader = DataLoader(trainset, batch_size=64, shuffle=False)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    # Load feature extraction model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights, progress=False).eval()
    model.fc = torch.nn.Identity()
    transform_2 = weights.transforms()
    # Extraction features
    features, targets = [], []
    for idx, batch in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            features.append(model(transform_2(batch[0])))
            targets.append(batch[1])
    for idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            features.append(model(transform_2(batch[0])))
            targets.append(batch[1])
    features = torch.concat(features)
    targets = torch.concat(targets)
    data_dict = {t: features[targets == t] for t in range(10)}
    return data_dict


def bag_collate_fn(batch_list):
    """
    Custom collate function for this dataset.
    """
    col_batch = {}
    for key in batch_list[0].keys():
        if key == "features":
            col_batch[key] = torch.concat([batch[key] for batch in batch_list])
        elif key == "sample_ids":
            col_batch[key] = {
                col: [batch[key][col] for batch in batch_list]
                for col in batch_list[0][key]
            }
        elif key == "targets":
            col_batch[key] = torch.stack([batch[key] for batch in batch_list])
        elif key == "numbers":
            col_batch[key] = torch.stack([batch[key] for batch in batch_list])
        elif key == "evidence":
            red_batch_list = []
            for batch in batch_list:
                red_batch_list.append({key: batch[key]})
            col_batch[key] = default_collate(red_batch_list)[key]
        else:
            col_batch[key] = torch.tensor([batch[key] for batch in batch_list])
    return col_batch


class NumberMILDataset(Dataset):

    def __init__(
        self,
        num_numbers,
        num_bags,
        num_instances,
        noise=1.0,
        threshold=1,
        sampling="hierarchical",
        features_type="mnist_resnet18",
        features_path=None,
    ):
        self.num_numbers = num_numbers
        self.num_bags = num_bags
        self.num_instances = num_instances
        self.thr = threshold
        self.numbers = []
        while len(self.numbers) < num_bags:
            if sampling == "unique":
                sampled_numbers = np.random.choice(
                    np.arange(self.num_numbers), size=num_instances, replace=False
                )
                self.numbers.append(sampled_numbers)
            elif sampling == "uniform":
                sampled_numbers = np.random.choice(
                    np.arange(self.num_numbers), size=num_instances, replace=True
                )
                self.numbers.append(sampled_numbers)
            elif sampling == "hierarchical":
                sampled_numbers = np.where(
                    np.random.randint(0, 2, size=num_numbers) == 1
                )[0]
                if len(sampled_numbers) > 0:
                    self.numbers.append(
                        np.random.choice(sampled_numbers, size=num_instances)
                    )
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling}")
        self.numbers = np.concatenate(self.numbers)
        if features_type == "onehot":
            self.features = (
                np.random.normal(
                    loc=0.0, scale=noise, size=(num_bags * num_instances, num_numbers)
                )
                + np.eye(num_numbers)[self.numbers]
            )
        elif features_type == "mnist_resnet18":
            data_dict = {
                idx: torch.load(os.path.join(features_path, f"class_{idx}.pt"))
                for idx in range(10)
            }
            self.features = np.stack(
                [
                    data_dict[n][np.random.choice(data_dict[n].shape[0])]
                    for n in self.numbers
                ]
            )
        else:
            raise ValueError(f"Unknown features type: {features_type}")

        self.numbers = torch.tensor(self.numbers.reshape(num_bags, num_instances))
        self.features = torch.tensor(
            self.features.reshape(num_bags, num_instances, -1), dtype=torch.float32
        )
        self.num_features = self.features.shape[-1]

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "bag_size": self.num_instances,
            "numbers": self.numbers[idx],
            "sample_ids": {},
        }


class SMILDataset(NumberMILDataset):

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        pos_number = 9
        item = super().__getitem__(idx)
        number_count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        if number_count[pos_number] >= self.thr:
            targets = torch.tensor([1])
        else:
            targets = torch.tensor([0])
        pos_evidence = (item["numbers"] == pos_number) * 1
        return {
            **item,
            "targets": targets,
            "evidence": {0: -pos_evidence, 1: pos_evidence},
        }


class PosNegDataset(NumberMILDataset):

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        pos_numbers = torch.tensor([4, 6, 8])
        neg_numbers = torch.tensor([5, 7, 9])
        item = super().__getitem__(idx)
        number_count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        if number_count[pos_numbers].sum() > sum(number_count[neg_numbers]):
            targets = torch.tensor([1])
        else:
            targets = torch.tensor([0])
        pos_evidence = torch.isin(item["numbers"], pos_numbers) * 1
        neg_evidence = torch.isin(item["numbers"], neg_numbers) * 1
        evidence = {0: neg_evidence - pos_evidence, 1: pos_evidence - neg_evidence}
        return {**item, "targets": targets, "evidence": evidence}


class FourBagsDataset(NumberMILDataset):

    @property
    def num_classes(self):
        return 4

    def __getitem__(self, idx):
        c1_number, c2_number = 8, 9
        item = super().__getitem__(idx)
        number_count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        if number_count[c1_number] >= self.thr > number_count[c2_number]:
            targets = torch.tensor([1])
        elif number_count[c2_number] >= self.thr > number_count[c1_number]:
            targets = torch.tensor([2])
        elif (
            number_count[c1_number] >= self.thr and number_count[c2_number] >= self.thr
        ):
            targets = torch.tensor([3])
        else:
            targets = torch.tensor([0])
        num_positions = {
            c1_number: (item["numbers"] == c1_number) * 1,
            c2_number: (item["numbers"] == c2_number) * 1,
        }
        evidence = {
            0: -num_positions[c1_number] - num_positions[c2_number],
            1: num_positions[c1_number] - num_positions[c2_number],
            2: -num_positions[c1_number] + num_positions[c2_number],
            3: num_positions[c1_number] + num_positions[c2_number],
        }
        return {**item, "targets": targets, "evidence": evidence}


class AdjacentPairsDataset(NumberMILDataset):

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        number_count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        evidence_thr = 5
        numbers = (number_count >= self.thr).nonzero().squeeze().tolist()
        pos_tuples = []
        if isinstance(numbers, list) and len(numbers) > 1:
            numbers = list(filter(lambda x: x < evidence_thr, numbers))
            for idx, num_0 in enumerate(numbers):
                num_1 = numbers[(idx + 1) % len(numbers)]
                if (num_0 + 1) == num_1:
                    pos_tuples.append([num_0, num_1])
        if len(pos_tuples) >= self.thr:
            targets = torch.tensor([1])
        else:
            targets = torch.tensor([0])
        pos_evidence = (
            torch.isin(item["numbers"], torch.tensor(pos_tuples).flatten())
        ) * 1
        return {
            **item,
            "targets": targets,
            "evidence": {0: -pos_evidence, 1: pos_evidence},
        }
