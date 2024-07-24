"""
Copyright (c) 2024 MPI-M, Clara Bayley


----- CLEO -----
File: awkward?handling.py
Project: pzSD
Created Date: Tuesday 24th July 2024
Author: Nils Niebaum (NN)
Additional Contributors:
-----
Last Modified: Tuesday 24th July 2024
Modified By: NN
-----
License: BSD 3-Clause "New" or "Revised" License
https://opensource.org/licenses/BSD-3-Clause
-----
File Description:
python class to handle awkwward array in a verz similar manner to xarray.
"""

# %%
import awkward as ak
from typing import Union, Tuple, Dict, List
from collections import Counter


class AwkwardDataArray:
    def __init__(self, name: str = "", data: ak.Array = ak.Array([])):
        self.name = name
        self.data = data

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @name.getter
    def name(self) -> str:
        return self._name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: ak.Array):
        self._data = data

    @data.getter
    def data(self) -> ak.Array:
        return self._data

    def __repr__(self):
        s = f"[\n{self.name}, {self.__class__.__name__} \n"
        s += f" - Data: {self.data}\n"
        s += f" - ID: {hex(id(self))}\n]"
        return s


class AwkwardDataset:
    def __init__(self, name: str = "", vars: Tuple[AwkwardDataArray] = ()):
        self.variables = vars
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @name.getter
    def name(self) -> str:
        return self._name

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(
        self, vars: Union[Tuple[AwkwardDataArray], List[AwkwardDataArray]]
    ) -> None:
        """
        This method sets the variables of the dataset.
        The variables are stored in the attribute variables.
        """
        # make sure to use the proper input type
        self._variables = self.__validate_variables_tuple__(vars)

    @variables.getter
    def variables(self) -> Tuple[AwkwardDataArray]:
        return self._variables

    def __validate_variables_tuple__(
        self, vars: Union[Tuple[AwkwardDataArray], List[AwkwardDataArray]]
    ) -> None:
        if isinstance(vars, tuple):
            pass
        elif isinstance(vars, list):
            vars = tuple(vars)
        else:
            raise TypeError(
                "vars must be a tuple or a list of AwkwardDataArray objects"
            )

        # validate that the variable names are unique
        names = self.__get_variables_names__(vars)
        # Get all values that occur more than once
        counts = Counter(names)
        non_unique_names = [item for item, count in counts.items() if count > 1]
        if len(non_unique_names) > 0:
            raise ValueError(
                f"Variable names must be unique. Error for {non_unique_names}"
            )
        else:
            return vars

    def __get_variables_names__(self, vars: Tuple[AwkwardDataArray]) -> Tuple[str]:
        """
        This method returns a tuple with the names of the variables.
        """
        return tuple(var.name for var in vars)

    def __get_variables_dict__(
        self, vars: Tuple[AwkwardDataArray]
    ) -> Dict[str, AwkwardDataArray]:
        """
        This method returns a dictionary with the data arrays as values and their names as keys.
        It can be used to access the data arrays by their names.
        """
        result = dict()
        for var in vars:
            result[var.name] = var
        return result

    def add_variable(self, var: AwkwardDataArray, overwrite: bool = True):
        """
        With this method, a variable can be added to the dataset.

        Parameters:
        var (AwkwardDataArray): The variable to be added to the dataset.
        overwrite (bool, optional): If True, the variable will be added even if it already exists in the dataset. Default is True.

        Raises:
        ValueError: If ``overwrite`` is ``False`` and the variable is already in the dataset.
        """
        vars_dict = self.__get_variables_dict__(self.variables)

        if var.name in vars_dict.keys() and overwrite is False:
            raise ValueError(f"Variable {var.name} already in dataset")
        else:
            vars_dict.update({var.name: var})
            self.variables = list(vars_dict.values())

    def drop_variable(self, name: str):
        """
        With this method, a variable can be removed from the dataset by providing its name.
        """
        vars_dict = self.__get_variables_dict__(self.variables)
        if name in vars_dict:
            vars_dict.drop(name)
            self.variables = tuple(vars_dict.values())
        else:
            raise ValueError(f"Variable {name} not in dataset")

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}\n"
        s += f"Name: {self.name}\n"
        s += f"Variables:\n{self.variables}"
        return s


# Test class

# Define classes and methods as shown above

# Example usage
data_array1 = AwkwardDataArray("data1")
data_array2 = AwkwardDataArray("data2")
dataset1 = AwkwardDataset(name="ds1", vars=(data_array1, data_array2))
dataset2 = AwkwardDataset(name="ds2", vars=(data_array1,))

print(dataset1)  # Output: {'data1': <AwkwardDataArray object>}
# print(dataset2)  # Output: {'data1': <AwkwardDataArray object>}
# print(data_array.datasets)  # Output: {<weakref at 0x7fb7dc5c5e10; to 'AwkwardDataset' at 0x7fb7dc5b78b0>, <weakref at 0x7fb7dc5c5f50; to 'AwkwardDataset' at 0x7fb7dc5b7e50>}

# Change the name of the data array
data_array1.name = "data_renamed"
data_array3 = AwkwardDataArray("data1")
data_array4 = AwkwardDataArray("data2")
dataset1.add_variable(data_array3)
dataset1.add_variable(data_array4)
print(dataset1)  # Output: {'data_renamed': <AwkwardDataArray object>}
# print(dataset2)  # Output: {'data_renamed': <AwkwardDataArray object>}
