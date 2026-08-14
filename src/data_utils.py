import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from src.pico.randaugment import RandomAugment
import copy


class WeaklySupervisedDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]

class ComparisonDataGenerator:
    def __init__(self, ground_truth_dataset, noise_type='clean', eta=0.0):
        self.dataset = ground_truth_dataset
        self.num_classes = len(self.dataset.classes)
        self.all_labels = np.arange(self.num_classes)
        self.original_data = self.dataset.data
        self.original_targets = torch.tensor(self.dataset.targets)
        self.noise_type = noise_type
        self.eta = eta

    def _apply_noise(self, candidate_set, true_label):
        if self.noise_type == 'noisy' and np.random.rand() < self.eta:
            # Remove true label from candidate set to introduce noise.
            candidate_set_mutable = set(candidate_set)
            if true_label in candidate_set_mutable:
                candidate_set_mutable.remove(true_label)
            
            # If the set becomes empty, add a random incorrect label.
            while not candidate_set_mutable:
                incorrect_labels = np.delete(self.all_labels, true_label)
                random_label = np.random.choice(incorrect_labels)
                candidate_set_mutable.add(random_label)

            return np.array(list(candidate_set_mutable))
        return candidate_set

    def generate_pl_dataset(self, k: int):
        if not 1 < k <= self.num_classes:
            raise ValueError(f"'k' must be between 2 and {self.num_classes}.")
        new_targets = []
        original_data = self.dataset.data
        for _, true_label in tqdm(self.dataset, desc="Processing PL"):
            incorrect_labels = np.delete(self.all_labels, true_label)
            num_to_select = k - 1
            additional_candidates = np.random.choice(
                incorrect_labels, size=num_to_select, replace=False
            )
            candidate_set = np.append(additional_candidates, true_label)
            
            # Apply noise if specified.
            candidate_set = self._apply_noise(candidate_set, true_label)

            candidate_set.sort()
            new_targets.append(torch.tensor(candidate_set))
        return WeaklySupervisedDataset(original_data, new_targets)

    def generate_pl_dataset_hierarchical(self, k: int, class_coarse):
        """Like generate_pl_dataset, but candidate labels are drawn
        preferentially from the same coarse superclass as the true label
        (the "CIFAR-100-H" hierarchical-ambiguity setting used by PiCO,
        Wang et al. ICLR 2022, Section 4.4 -- see docs/pico_explanation.md).
        Harder than uniform sampling since same-superclass classes are
        visually similar.

        NOTE (2026-08-14, confirmed via real PDF text extraction): the
        paper's actual generation process is q-based (each same-superclass
        false label independently included with probability q), NOT this
        fixed-k selection -- see generate_pl_dataset_hierarchical_variable
        below for the literal reproduction. This k-based method is an
        adaptation of the paper's core idea (same-superclass restriction) to
        fit this repo's k-swept CLI, kept because it's still a meaningful,
        harder-than-uniform PL generation mode; --dataset cifar100-h uses
        the q-based method below by default (paper-faithful), not this one.

        class_coarse[c] = coarse superclass id (0..19) of (remapped) fine
        class c. Ground truth: extracted directly from CIFAR-100's own
        pickled 'coarse_labels' field (see src/cifar100_subset.py's
        _CIFAR100_FINE_TO_COARSE), not reconstructed from memory.

        If the true label's superclass doesn't contain enough OTHER
        (selected) sibling classes to fill k-1 candidates -- which happens
        whenever a C-class subset selection didn't happen to include most of
        that superclass's members -- falls back to filling the remainder
        uniformly from the other classes. This is logged, not silent.
        """
        if not 1 < k <= self.num_classes:
            raise ValueError(f"'k' must be between 2 and {self.num_classes}.")
        class_coarse = np.asarray(class_coarse)
        new_targets = []
        fallback_count = 0
        for _, true_label in tqdm(self.dataset, desc="Processing PL (hierarchical)"):
            true_label = int(true_label)
            same_superclass = np.where(class_coarse == class_coarse[true_label])[0]
            same_superclass = same_superclass[same_superclass != true_label]
            num_to_select = k - 1
            if len(same_superclass) >= num_to_select:
                additional_candidates = np.random.choice(same_superclass, size=num_to_select, replace=False)
            else:
                fallback_count += 1
                other = np.delete(self.all_labels, true_label)
                remaining_pool = np.setdiff1d(other, same_superclass)
                extra_needed = num_to_select - len(same_superclass)
                extra = np.random.choice(remaining_pool, size=extra_needed, replace=False)
                additional_candidates = np.concatenate([same_superclass, extra])

            candidate_set = np.append(additional_candidates, true_label)
            candidate_set = self._apply_noise(candidate_set, true_label)
            candidate_set.sort()
            new_targets.append(torch.tensor(candidate_set))

        if fallback_count:
            print(f"  [cifar100-h] {fallback_count}/{len(new_targets)} samples fell back to "
                  f"(partial) uniform sampling -- their coarse superclass didn't have enough "
                  f"selected sibling classes to fill k-1 hierarchical candidates.", flush=True)
        return WeaklySupervisedDataset(self.dataset.data, new_targets)

    def generate_pl_dataset_hierarchical_variable(self, q: float, class_coarse):
        """The paper-exact CIFAR-100-H generation process. Quoted directly
        from PiCO (Wang et al., ICLR 2022), Section 4.4 (confirmed 2026-08-14
        via a real PDF text extraction, not reconstructed from memory or a
        secondary summary): "CIFAR-100 with hierarchical labels (CIFAR-100-H),
        where we generate candidate labels that belong to the same
        superclass. We set q=0.5 for CIFAR-100 with hierarchical labels"
        (Table 6 additionally reports q in {0.1, 0.5, 0.8}).

        This is the same *mechanism* as generate_variable_pl_cl_datasets
        (each false label independently included with probability q, not a
        fixed candidate-set size k) -- generate_pl_dataset_hierarchical
        above (the one actually wired into this pipeline's --dataset
        cifar100-h, via prepare_cifar100_subset's `hierarchical=` flag) uses
        a fixed-k selection instead, to fit this repo's C-x-k sweep CLI. That
        makes generate_pl_dataset_hierarchical an adaptation of the paper's
        idea (same-superclass restriction), not a literal reproduction of
        its generation process -- THIS method is the literal reproduction.
        Wired into --dataset cifar100-h via prepare_cifar100_subset's
        `hierarchical_q=` flag (mutually exclusive with `hierarchical=`),
        which bypasses the k-sweep for a single q-based cell (DatasetSpec.
        sweeps_k=False), same mechanism as pre-ambiguous datasets' single-
        cell treatment in runner.py.
        """
        if not 0 <= q <= 1:
            raise ValueError("'q' must be between 0 and 1.")
        class_coarse = np.asarray(class_coarse)
        new_targets = []
        for _, true_label in tqdm(self.dataset, desc="Processing PL (hierarchical, q-based)"):
            true_label = int(true_label)
            pl_set = {true_label}
            same_superclass = np.where(class_coarse == class_coarse[true_label])[0]
            same_superclass = same_superclass[same_superclass != true_label]
            for label in same_superclass:
                if np.random.rand() < q:
                    pl_set.add(int(label))
            candidate_set = np.array(sorted(pl_set))
            candidate_set = self._apply_noise(candidate_set, true_label)
            candidate_set.sort()
            new_targets.append(torch.tensor(candidate_set, dtype=torch.long))
        return WeaklySupervisedDataset(self.dataset.data, new_targets)

    def generate_cl_dataset(self, m: int):
        if not 0 < m < self.num_classes:
            raise ValueError(f"'m' must be between 1 and {self.num_classes - 1}.")
        new_targets = []
        original_data = self.dataset.data
        for _, true_label in tqdm(self.dataset, desc="Processing CL"):
            incorrect_labels = np.delete(self.all_labels, true_label)
            complementary_set = np.random.choice(
                incorrect_labels, size=m, replace=False
            )
            complementary_set.sort()
            new_targets.append(torch.tensor(complementary_set))
        return WeaklySupervisedDataset(original_data, new_targets)

    def generate_variable_pl_cl_datasets(self, q: float, num_classes: int):
        if not 0 <= q <= 1:
            raise ValueError("'q' must be between 0 and 1.")

        pl_targets = []
        cl_targets = []
        original_data = self.dataset.data
        all_labels = np.arange(num_classes)

        for _, true_label in tqdm(self.dataset, desc="Processing Variable PL/CL"):
            # Generate PL dataset
            pl_set = {true_label}
            false_labels = np.delete(all_labels, true_label)
            for label in false_labels:
                if np.random.rand() < q:
                    pl_set.add(label)
            
            # Apply noise if specified.
            pl_set_array = self._apply_noise(np.array(list(pl_set)), true_label)
            pl_target = sorted(list(pl_set_array))
            pl_targets.append(torch.tensor(pl_target, dtype=torch.long))

            # Generate CL dataset from the (potentially noisy) PL set.
            cl_set = set(all_labels) - set(pl_target)
            cl_target = sorted(list(cl_set))
            cl_targets.append(torch.tensor(cl_target, dtype=torch.long))

        pl_dataset = WeaklySupervisedDataset(original_data, pl_targets)
        cl_dataset = WeaklySupervisedDataset(original_data, cl_targets)

        return pl_dataset, cl_dataset

class PicoDataset(Dataset):

    def __init__(self, pl_dataset_raw, original_labels, image_size=32,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616)):
        self.images = pl_dataset_raw.data
        self.given_label_matrix_sparse = pl_dataset_raw.targets
        self.true_labels = original_labels

        self.num_classes = len(set(original_labels.numpy()))

        # Weak and strong augmentations for contrastive learning.
        # image_size/mean/std default to the original CIFAR values so existing
        # call sites (no args passed) are unaffected. RGB-only ops (ColorJitter,
        # RandomGrayscale) mean this class should only be used for 3-channel
        # image datasets -- see DatasetSpec.supports_pico_family.
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(n=3, m=5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        image = self.images[index]
        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        
        # Create one-hot encoded partial label vector.
        p_label = self.given_label_matrix_sparse[index]
        each_label = torch.zeros(self.num_classes, dtype=torch.float)
        each_label[p_label] = 1.0
        
        each_true_label = self.true_labels[index]
        return each_image_w, each_image_s, each_label, each_true_label, index

class ComCoDataset(Dataset):
    """Dataset for ComCo: returns weak/strong augmented pairs with complementary label masks."""

    def __init__(self, cl_dataset_raw, original_labels, image_size=32,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616)):
        self.images = cl_dataset_raw.data
        self.true_labels = original_labels
        self.num_classes = len(set(original_labels.numpy()))

        # Pre-build dense binary complementary masks [C] from variable-length sparse labels
        self.comp_masks = []
        for cl_labels in cl_dataset_raw.targets:
            mask = torch.zeros(self.num_classes, dtype=torch.float)
            mask[cl_labels] = 1.0
            self.comp_masks.append(mask)

        # image_size/mean/std default to the original CIFAR values (see PicoDataset
        # for why this class should only be used for 3-channel image datasets).
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(n=3, m=5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image = self.images[index]
        img_w = self.weak_transform(image)
        img_s = self.strong_transform(image)
        comp_mask = self.comp_masks[index]
        true_label = self.true_labels[index]
        return img_w, img_s, comp_mask, true_label, index


class SoLarDataset(Dataset):
    def __init__(self, pl_dataset_raw, original_labels, image_size=32,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)):
        self.images = pl_dataset_raw.data
        self.given_label_matrix_sparse = pl_dataset_raw.targets
        self.true_labels = original_labels

        self.num_classes = len(set(original_labels.numpy()))

        self.weak_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(image_size, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        self.strong_transform = copy.deepcopy(self.weak_transform)
        self.strong_transform.transforms.insert(1, RandomAugment(3,5))

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        image = self.images[index]
        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)

        # Create one-hot encoded partial label vector.
        p_label = self.given_label_matrix_sparse[index]
        each_label = torch.zeros(self.num_classes, dtype=torch.float)
        each_label[p_label] = 1.0
        
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index