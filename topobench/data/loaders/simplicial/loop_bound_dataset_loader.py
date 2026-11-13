"""Loader for the LoopBound diagnostic dataset."""

import os

from topobench.data.datasets.loop_bound_dataset import LoopBoundDataset
from topobench.data.loaders.base import AbstractLoader


class LoopBoundDatasetLoader(AbstractLoader):
    """Load LoopBound (marked edge cycle bounds or not).

    Parameters
    ----------
    parameters : DictConfig
        Loader parameters with ``data_dir`` and ``data_name``.
    **kwargs : dict
        Ignored.
    """

    def __init__(self, parameters, **kwargs):
        super().__init__(parameters)

    def load_dataset(self, **kwargs):
        """Instantiate the dataset.

        Parameters
        ----------
        **kwargs : dict
            Ignored.

        Returns
        -------
        LoopBoundDataset
            The dataset.
        """
        return LoopBoundDataset(
            os.path.join(self.root_data_dir, self.parameters.data_name)
        )

    def load(self, **kwargs):
        """Load the dataset and its processed directory.

        Parameters
        ----------
        **kwargs : dict
            Ignored.

        Returns
        -------
        tuple
            ``(dataset, processed_dir)``.
        """
        dataset = self.load_dataset()
        return dataset, dataset.processed_dir
