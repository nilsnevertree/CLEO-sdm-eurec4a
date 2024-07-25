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
import numpy as np
from typing import Union, Tuple, Dict, List
from collections import Counter
import xarray as xr
from pySD.sdmout_src import sdtracing


def get_awkward_shape(a: ak.highlevel.Array):
    """
    Get the shape of the awkward array a as a list.
    Variable axis lengths are replaced by ``np.nan``.

    Parameters
    ----------
    a : ak.Array
        The input array.

    Returns
    -------
    list
        The shape of the array as a list.
        ``var`` positions are replaced by ``np.nan``.
    """

    # check for number of dimensions
    ndim = a.ndim
    # create output list
    shape = []
    # For each dinemsion, get the number of elements.
    # If the number of elements changes over the axis, np.nan is used to indicate a variable axis length
    for dim in range(ndim):
        num = ak.num(a, axis=dim)
        # for the 0th axis, the number of elements is an integer
        if isinstance(num, np.ndarray):
            num = int(num)
            shape.append(num)
        else:
            maxi = int(ak.max(num))
            mini = int(ak.min(num))
            if maxi == mini:
                shape.append(maxi)
            else:
                shape.append(np.nan)
    return shape


def assert_same_shape(a: ak.highlevel.Array, b: ak.highlevel.Array):
    """
    Assert that the two awkward arrays have the same shape.
    Variable axis lengths are replaced by ``np.nan``.

    Parameters
    ----------
    a : ak.Array
        The first input array.
    b : ak.Array
        The second input array.

    Raises
    ------
    ValueError
        If the shapes of the two arrays do not match.
    """

    shape_a = get_awkward_shape(a)
    shape_b = get_awkward_shape(b)
    if shape_a != shape_b:
        raise ValueError(
            f"The shapes of the two arrays do not match: {shape_a} != {shape_b}"
        )


def assert_only_last_axis_variable(a: ak.highlevel.Array):
    """
    Assert that the awkward array has only variable axis lengths at the last axis.

    Parameters
    ----------
    a : ak.Array
        The input array.

    Raises
    ------
    ValueError
        If the array has variable axis lengths at other axes than the last axis.
    """

    shape = get_awkward_shape(a)
    if any([np.isnan(elem) for elem in shape[:-1]]):
        raise ValueError(
            "The array has variable axis lengths at other axes than the last axis."
        )


class AwkwardDataArray:
    def __init__(
        self,
        data: Union[np.ndarray, ak.Array, xr.DataArray] = ak.Array([]),
        name: str = "",
        units: str = "",
        metadata: Dict = dict(),
    ):
        """
        Class to handle awkward array in a very similar manner to xarray DataArray

        Parameters
        ----------
        name : str, optional
            The name of the data array.
        data : np.ndarray or ak.Array or xr.DataArray optional
            The data of the data array.
            If a xr.DataArray is provided, the data is extracted from the DataArray.
            Metadata will be constructed directly from attrs and ``units`` will be set to units key from attrs if existing, otherwise no units.
            Default is an empty ak.Array.
        units : str, optional
            The units of the data array.
            Default is an empty string.
        metadata : dict, optional
            The metadata of the data array.
            Default is an empty dictionary.

        Attributes
        ----------
        name : str
            The name of the data array.
        data : ak.Array
            The data of the data array.
        metadata : dict
            The metadata of the data array.

        """

        if isinstance(data, xr.DataArray):
            self.__from_xarrayDataArray__(data)

        else:
            self.name = name
            self.data = data
            self.units = units
            self.metadata = metadata

    # name property
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        if isinstance(name, str):
            self._name = name
        elif name is None:
            self._name = ""
        else:
            raise TypeError("name must be a string")

    @name.getter
    def name(self) -> str:
        return self._name

    # data property
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: Union[np.ndarray, ak.Array]):
        if isinstance(data, np.ndarray):
            self._data = ak.Array(data)
        elif isinstance(data, ak.Array):
            self._data = data
        else:
            raise TypeError("data must be an ak.Array or np.ndarray")

    @data.getter
    def data(self) -> ak.Array:
        return self._data

    # units property
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units: str):
        self._units = units

    @units.getter
    def units(self) -> str:
        return self._units

    # metadata property
    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        self._metadata = metadata

    @metadata.getter
    def metadata(self) -> dict:
        return self._metadata

    # representation of the class
    def __str__(self):
        s = f"{self.name} ({self.__class__.__name__}):\n"
        s += f"- data typestr: {self.data.typestr}\n"
        s += f"- data: {self.data}\n"
        s += f"- id: {hex(id(self))}\n"
        s += f"- units: {self.units}\n"
        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (name: {self.name})"

    # Add methods
    def _add_(self, other) -> "AwkwardDataArray":
        """
        This method overloads the + operator.
        It performs element-wise addition of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be added to the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the addition.
            The same name, metadata and units are used.
        """
        if isinstance(other, AwkwardDataArray):
            new_data = self.data + other.data
        else:
            new_data = self.data + other

        return AwkwardDataArray(
            name=self.name,
            data=new_data,
            metadata=self.metadata,
            units=self.units,
        )

    def __add__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the + operator.
        It performs element-wise addition of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be added to the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the addition.
            The same name, metadata and units are used.
        """
        print("add")
        return self._add_(other)

    def __radd__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the + operator.
        It performs element-wise addition of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be added to the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the addition.
            The same name, metadata and units are used.
        """
        print("radd")
        return self._add_(other)

    # Add multiplication method
    def _mul_(self, other) -> "AwkwardDataArray":
        """
        This method overloads the * operator.
        It performs element-wise multiplication of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be multiplied with the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the multiplication.
            The same name, metadata and units are used.
        """
        if isinstance(other, AwkwardDataArray):
            new_data = self.data * other.data
            new_units = self.units + "*" + other.units
        else:
            new_data = self.data * other
            new_units = self.units

        return AwkwardDataArray(
            name=self.name,
            data=new_data,
            metadata=self.metadata,
            units=new_units,
        )

    def __mul__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the * operator.
        It performs element-wise multiplication of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be multiplied with the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the multiplication.
            The same name, metadata and units are used.
        """
        print("add")
        return self._mul_(other)

    def __rmul__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the * operator.
        It performs element-wise multiplication of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be multiplied with the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the multiplication.
            The same name, metadata and units are used.
        """
        print("radd")
        return self._mul_(other)

    # Add subtraction method
    def _sub_(self, other) -> "AwkwardDataArray":
        """
        This method overloads the - operator.
        It performs element-wise subtraction of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be subtracted from the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the subtraction.
            The same name, metadata and units are used.
        """
        if isinstance(other, AwkwardDataArray):
            new_data = self.data - other.data
        else:
            new_data = self.data - other

        return AwkwardDataArray(
            name=self.name,
            data=new_data,
            metadata=self.metadata,
            units=self.units,
        )

    def __sub__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the - operator.
        It performs element-wise subtraction of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be subtracted from the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the subtraction.
            The same name, metadata and units are used.
        """
        print("add")
        return self._sub_(other)

    def __rsub__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the - operator.
        It performs element-wise subtraction of the data attribute with another object.

        Parameters
        ----------
        other : object
            The object to be subtracted from the data attribute.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the subtraction.
            The same name, metadata and units are used.
        """
        print("radd")
        return self._sub_(other)

    # Add division method
    def __truediv__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the / operator.
        It performs element-wise division of the data attribute by another object.

        Parameters
        ----------
        other : object
            The object to divide the data attribute by.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the division.
            The same name, metadata and units are used.
        """
        print("add")
        if isinstance(other, AwkwardDataArray):
            new_data = self.data / other.data
            new_units = self.units + "/" + other.units
        else:
            new_data = self.data / other
            new_units = self.units

        return AwkwardDataArray(
            name=self.name,
            data=new_data,
            metadata=self.metadata,
            units=new_units,
        )

        return self._truediv_(other)

    def __rtruediv__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the / operator.
        It performs element-wise division of the data attribute by another object.

        Parameters
        ----------
        other : object
            The object to divide the data attribute by.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the division.
            The same name, metadata and units are used.
        """
        print("radd")
        if isinstance(other, AwkwardDataArray):
            new_data = self.data / other.data
            new_units = other.units + "/" + self.units
        else:
            new_data = self.data / other
            new_units = self.units

        return AwkwardDataArray(
            name=self.name,
            data=new_data,
            metadata=self.metadata,
            units=new_units,
        )

    def __pow__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the ** operator.
        It performs element-wise exponentiation of the data attribute by another object.

        Parameters
        ----------
        other : object
            The object to exponentiate the data attribute by.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the exponentiation.
            The same name, metadata and units are used.
        """
        print("add")
        if isinstance(other, AwkwardDataArray):
            new_data = self.data**other.data
            new_units = self.units + "^" + other.units
        else:
            new_data = self.data**other
            new_units = self.units

        return AwkwardDataArray(
            name=self.name,
            data=new_data,
            metadata=self.metadata,
            units=new_units,
        )

    def __rpow__(self, other) -> "AwkwardDataArray":
        """
        This method overloads the ** operator.
        It performs element-wise exponentiation of the data attribute by another object.

        Parameters
        ----------
        other : object
            The object to exponentiate the data attribute by.

        Returns
        -------
        AwkwardDataArray
            A new AwkwardDataArray object with the result of the exponentiation.
            The same name, metadata and units are used.
        """
        print("radd")
        if isinstance(other, AwkwardDataArray):
            new_data = other.data**self.data
            new_units = other.units + "^" + self.units
        else:
            new_data = other**self.data
            new_units = self.units

        return AwkwardDataArray(
            name=self.name,
            data=new_data,
            metadata=self.metadata,
            units=new_units,
        )

    def flatten(self):
        """
        This function flattens the data of the attribute.
        The data is flattened and stored in the attribute data.
        It is a mutating function!
        """
        self.data = ak.flatten(self.data, axis=None)

    def sort_by(self, sort_array: ak.Array) -> None:
        """
        This function sorts the attribute by a sort array.
        The attribute is sorted by a sort array by creating a new attribute with the sorted data.

        Parameters
        ----------
        self : SupersAttribute
            The attribute to be sorted.
        sort_array : ak.Array
            The array to sort the attribute by.
        """

        # sort the data
        sorted_data = self.data[sort_array]

        self.data = sorted_data

    def __from_xarrayDataArray__(self, da: xr.DataArray):
        """
        This function converts an xarray DataArray to a AwkwardDataArray.
        """
        self.name = da.name
        self.data = da.data
        try:
            self.metadata = da.attrs
        except AttributeError:
            self.metadata = dict()
        try:
            self.units = self.metadata.pop("units")
        except KeyError:
            self.units = ""


class AwkwardCoord(AwkwardDataArray):
    """
    This class is a child of the AwkwardDataArray class.
    It should replace the xarray coordinate setup.
    For this, the class inherits from the AwkwardDataArray class.

    """

    def __init__(
        self,
        data: Union[np.ndarray, ak.Array, xr.DataArray] = ak.Array([]),
        name: str = "",
        units: str = "",
        metadata: Dict = dict(),
    ):
        """
        Class to handle awkward array in a very similar manner to xarray DataArray

        Parameters
        ----------
        name : str, optional
            The name of the data array.
        data : np.ndarray or ak.Array or xr.DataArray optional
            The data of the data array.
            If a xr.DataArray is provided, the data is extracted from the DataArray.
            Metadata will be constructed directly from attrs and ``units`` will be set to units key from attrs if existing, otherwise no units.
            Default is an empty ak.Array.
        units : str, optional
            The units of the data array.
            Default is an empty string.
        metadata : dict, optional
            The metadata of the data array.
            Default is an empty dictionary.

        Attributes
        ----------
        name : str
            The name of the data array.
        data : ak.Array
            The data of the data array.
        metadata : dict
            The metadata of the data array.

        """
        super().__init__(name=name, data=data, units=units, metadata=metadata)

    # the dimensino is a 1D array
    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim: int):
        self._dim = dim

    @dim.getter
    def dim(self) -> int:
        return self._dim

    # The coordinate dimension
    @property
    def coord(self):
        return self._coord

    @coord.getter
    def coord(self) -> ak.Array:
        return self._coord

    def __create_coord__(self, right: bool = False) -> ak.Array:
        """
        With this method, the coordinate array is created for the AwkwardCoord.
        It uses the given 1D dimension array ``self.dim`` to create the coordinate array.
        For this, the numpy.digitize function is used to digitize the data array.
        The bins used for this are given by ``self.dim``.

        Parameters
        ----------
        right : bool, optional
            As from the numpy documentation:
            Indicating whether the intervals include the right or the left bin edge.
            Default behavior is (right==False) indicating that the interval does not include the right edge.

        """

        data_ndim = self.data.ndim

        if data_ndim == 1:
            coord = np.digitize(x=self.data, bins=self.dim, right=right)
        elif data_ndim == 2:
            coord = sdtracing.ak_digitize_2D(x=self.data, bins=self.dim, right=False)
        elif self.data.ndim == 3:
            coord = sdtracing.ak_digitize_3D(x=self.data, bins=self.dim, right=False)
        else:
            raise NotImplementedError(
                "Only 1D, 2D and 3D arrays are supported for digitization till now."
            )

        coord = self.__validate_coord__(coord)
        return coord

    def __validate_coord__(self, coord: ak.Array) -> ak.Array:
        """
        This method validates the coord.
        The coordinate must be an ak.Array with integer values.
        It needs to be of the same shape as the data.

        Parameters
        ----------
        coord : ak.Array
            The coord to be validated.

        Returns
        -------
        ak.Array
            The validated coord.
        """
        dtype = ak.flatten(coord, axis=None).type.content
        if not np.issubdtype(dtype, np.integer):
            raise TypeError(
                "Coordinate must be an ak.Array with integer values. If you want to use individual data values as coordinate, please use the approriate class for this."
            )
        data_shape = get_awkward_shape(self.data)
        coord_shape = get_awkward_shape(coord)
        if data_shape != coord_shape:
            raise AssertionError("Coordinate must have the same shape as the data.")

        return


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
        # validate type of vars
        for var in vars:
            if not isinstance(var, AwkwardDataArray):
                raise TypeError(
                    "vars must be a tuple or a list of AwkwardDataArray objects."
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

    def __str__(self) -> str:
        s = f"{self.name} ({self.__class__.__name__})\n"
        s += "Variables:\n--------------\n"
        for var in self.variables:
            s += f"{var}\n"
        return s

    def __repr__(self) -> str:
        # return f"{self.__class__.__name__} (name: {self.name})"
        return self.__str__()

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
            vars_dict.pop(name)
            self.variables = tuple(vars_dict.values())
        else:
            raise ValueError(f"Variable {name} not in dataset")

    def flatten(self):
        """
        This function flattens the data of all variables in the Dataset.
        this mutates the DataArrays corresponding to the variables.
        """
        for var in self.variables:
            var.flatten()
        # TODO: make sure to erase the coordinates
        #    self.coordinates(coords=[])


# Define classes and methods as shown above

# Example usage
data_array1 = AwkwardDataArray(name="data1")
data_array2 = AwkwardDataArray(name="data2")
dataset1 = AwkwardDataset(name="ds1", vars=(data_array1, data_array2))
dataset2 = AwkwardDataset(name="ds2", vars=(data_array1,))

print(dataset1)  # Output: {'data1': <AwkwardDataArray object>}
# print(dataset2)  # Output: {'data1': <AwkwardDataArray object>}
# print(data_array.datasets)  # Output: {<weakref at 0x7fb7dc5c5e10; to 'AwkwardDataset' at 0x7fb7dc5b78b0>, <weakref at 0x7fb7dc5c5f50; to 'AwkwardDataset' at 0x7fb7dc5b7e50>}

# Change the name of the data array
data_array1.name = "data_renamed"
data_array3 = AwkwardDataArray(name="data1")
data_array4 = AwkwardDataArray(name="data2")
dataset1.add_variable(data_array3)
dataset1.add_variable(data_array4)
# print(dataset1)  # Output: {'data_renamed': <AwkwardDataArray object>}
# print(dataset2)  # Output: {'data_renamed': <AwkwardDataArray object>}


data = ak.Array(
    [
        [10.0, 20, 30],
        [12, 14],
        [40, 50],
        [90],
    ]
)
time_values = ak.Array(
    [
        0,
        1,
        2,
        3,
    ]
)

time = data * 0 + time_values

gridbox = ak.Array(
    [
        [0, 0, 0],
        [1, 0],
        [1, 2],
        [2],
    ]
)

sdId = ak.Array(
    [
        [3, 1, 2],
        [2, 1],
        [1, 3],
        [7],
    ]
)

radius = ak.Array(
    [
        [1.1, 2.6, 2.5],
        [2.1, 3.5],
        [1.2, 3.4],
        [3.1],
    ]
)

bins = np.arange(0, 4, 1)


time = AwkwardDataArray(time, "time", units="s")
data = AwkwardDataArray(data, "data", units="m/s")
gridbox = AwkwardDataArray(gridbox, "gridbox", units="")
sdId = AwkwardDataArray(sdId, "sdId", units="")
radius = AwkwardDataArray(radius, "radius", units="m")

ds = AwkwardDataset("dataset", (time, data, gridbox, sdId, radius))

data + data
# %%
