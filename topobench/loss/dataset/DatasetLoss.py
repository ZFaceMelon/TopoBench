"""Loss module for the topobench package."""

import torch
import torch_geometric

from topobench.loss.base import AbstractLoss


class DatasetLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing the dataset loss information. An optional
        ``class_weights`` entry (cross-entropy classification only) is either
        a list of per-class weights or ``"balanced"``: inverse class
        frequency of the training labels, set via ``set_class_weights_from``.
    """

    def __init__(self, dataset_loss):
        super().__init__()
        self.task = dataset_loss["task"]
        self.loss_type = dataset_loss["loss_type"]
        weights = dataset_loss.get("class_weights", None)
        self.class_weights = None
        self.needs_class_weights = False
        if weights is not None:
            if self.task != "classification" or (
                self.loss_type != "cross_entropy"
            ):
                raise ValueError(
                    "class_weights needs cross_entropy classification"
                )
            if weights == "balanced":
                self.needs_class_weights = True
            else:
                self.class_weights = torch.tensor(
                    [float(w) for w in weights]
                )
        # Dataset loss
        if self.task == "classification":
            if self.loss_type == "cross_entropy":
                self.criterion = torch.nn.CrossEntropyLoss()
            elif self.loss_type == "BCE":
                self.criterion = torch.nn.BCEWithLogitsLoss()
            else:
                raise ValueError(
                    "Classification loss must be cross_entropy or BCE"
                )
        elif self.task == "multilabel classification":
            assert self.loss_type == "BCE", (
                "Invalid loss type for classification task,TB supports only BCE for multilabel classification task"
            )
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif (
            self.task == "regression"
            and self.loss_type == "mse"
            or (
                self.task == "multioutput classification"
                and self.loss_type == "mse"
            )
        ):
            self.criterion = torch.nn.MSELoss()

        elif self.task == "regression" and self.loss_type == "mae":
            self.criterion = torch.nn.L1Loss()
        else:
            raise Exception("Loss is not defined")

    def set_class_weights_from(self, labels, num_classes=None):
        r"""Set balanced weights ``n / (classes * count_c)`` from labels.

        Parameters
        ----------
        labels : torch.Tensor
            Training labels.
        num_classes : int, optional
            Number of classes; inferred from the labels if None
            (default: None).
        """
        labels = labels.reshape(-1).long()
        counts = torch.bincount(labels, minlength=num_classes or 0).double()
        weights = labels.numel() / (len(counts) * counts.clamp_min(1))
        self.class_weights = weights.float()
        self.needs_class_weights = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, loss_type={self.loss_type})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        logits = model_out["logits"]
        target = model_out["labels"]

        return self.forward_criterion(logits, target)

    def forward_criterion(self, logits, target):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        if self.task == "regression":
            target = target.reshape_as(logits)
            dataset_loss = self.criterion(logits, target)

        elif self.task == "multioutput classification":
            dataset_loss = self.criterion(logits, target.float())

        elif self.task == "classification":
            if self.loss_type == "BCE":
                target = target.to(logits).reshape_as(logits)
            if self.needs_class_weights:
                raise RuntimeError(
                    "Balanced class weights were requested but never set"
                )
            if self.class_weights is not None:
                dataset_loss = torch.nn.functional.cross_entropy(
                    logits,
                    target,
                    weight=self.class_weights.to(logits.device, logits.dtype),
                )
            else:
                dataset_loss = self.criterion(logits, target)

        elif self.task == "multilabel classification":
            mask = ~torch.isnan(target)
            # Avoid NaN values in the target
            target = torch.where(mask, target, torch.zeros_like(target))
            loss = self.criterion(logits, target.float())
            # Mask out the loss for NaN values
            loss = loss * mask
            # Take out average
            valid_rows = mask.any(dim=-1)
            row_loss = loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            dataset_loss = (
                row_loss[valid_rows].mean()
                if valid_rows.any()
                else logits.sum() * 0
            )

        else:
            raise Exception("Loss is not defined")

        return dataset_loss
