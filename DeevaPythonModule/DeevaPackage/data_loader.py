from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass(frozen=True)
class Dataset:
    """Class to represent a dataset."""
    tr: np.array
    tr_y: np.array
    validation: np.array
    validation_y: np.array
    test: np.array
    test_y: np.array

    def __post_init__(self):
        """
        Constructor to convert all data arrays to contiguous arrays as required by C++
        """

        assert len(self.tr_y.shape) == 1, "Labels (tr_y) must be a 1D array."
        assert len(self.validation_y.shape) == 1, "Labels (validation_y) must be a 1D array."
        assert len(self.test_y.shape) == 1, "Labels (test_y) must be a 1D array."

        self.tr = self._validate_and_convert(self.tr, 'tr')
        self.validation = self._validate_and_convert(self.validation, 'validation')
        self.test = self._validate_and_convert(self.test, 'test')

          # Ensure the number of rows in tr is the same as the length of tr_y
        assert self.tr.shape[0] == len(self.tr_y), "Number of rows in tr must be the same as the length of tr_y."
        assert self.validation.shape[0] == len(self.validation_y), "Number of rows in validation must be the same as the length of validation_y."
        assert self.test.shape[0] == len(self.test_y), "Number of rows in test must be the same as the length of test_y."




    def _validate_and_convert(self, array, name):
        """
        Validate that the array is a 2D array, and convert it to a contiguous array.

        Parameters:
        - array: numpy array, input array
        - name: str, name of the array for error messages

        Returns:
        - numpy array: the validated and converted array
        """
        assert len(array.shape) == 2, f"{name} must be a 2D array."
        return np.ascontiguousarray(array)


class DatasetLoader:
    def __init__(
        self,
        training_data: np.array,
        training_labels: np.array,
        validation_data: Optional[np.array] = None,
        validation_labels: Optional[np.array] = None,
        test_data: Optional[np.array] = np.empty([0,0]),
        test_labels: Optional[np.array] = np.empty([0]),
        random_split: bool = False,
    ):
        """
        Initialize the DatasetLoader.

        Parameters:
        - training_data: numpy array, training dataset
        - training_labels: numpy array, labels for the training dataset
        - validation_data: numpy array, optional validation dataset
        - validation_labels: numpy array, optional labels for the validation dataset
        - test_data: numpy array, optional test dataset
        - test_labels: numpy array, optional labels for the test dataset
        - random_split: bool, whether to perform a random split for training and validation
        """
        if random_split:
            self._load_random_split(training_data, training_labels, validation_data) # training split into new traning and validation
        else:
            if validation_data is None or validation_labels is None: # training == validation
                self._dataset = Dataset(
                    tr=training_data,
                    tr_y=training_labels,
                    validation=training_data,
                    validation_y=training_labels,
                    test=test_data,
                    test_y=test_labels,
                )
            else:
                self._dataset = Dataset( # validation given
                    tr=training_data,
                    tr_y=training_labels,
                    validation=validation_data,
                    validation_y=validation_labels,
                    test=test_data,
                    test_y=test_labels,
                )

    @property
    def dataset(self) -> Dataset:
        """
        Get the read-only dataset property.

        Returns:
        - Dataset: The loaded dataset.
        """
        return self._dataset

    def _load_random_split(
        self,
        training_data: np.array,
        training_labels: np.array,
    ):
        """
        Load dataset with a random split for training and validation.

        Parameters:
        - training_data: numpy array, training dataset
        - training_labels: numpy array, labels for the training dataset
        - validation_data: numpy array, optional validation dataset
        """
        if training_data is None or training_labels is None:
            raise ValueError("Random split requires training data and labels.")

        # Randomly split training data into training and validation sets
        indices = np.random.permutation(len(training_data))
        split_index = int(0.8 * len(indices))

        tr_indices, val_indices = indices[:split_index], indices[split_index:]

        self._dataset = Dataset(
            tr=training_data[tr_indices],
            tr_y=training_labels[tr_indices],
            validation=training_data[val_indices],
            validation_y=training_labels[val_indices],
            test=np.empty([0,0]),
            test_y=np.empty([0]),
        )

    

