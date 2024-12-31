# Imports ----
import numpy as np
import scipy.sparse as sp
import polars as pl
from turgnamh.utils.printing import b
from turgnamh.settings import logger

# Typing ----
from typing import get_args, Union
LayerData = Union[np.ndarray, sp.sparray]


# Classes ----
class Layer:
    data: LayerData
    obs_names: pl.Series
    feat_names: pl.Series
    
    
    # Constructors ----
    def __init__(
            self, 
            data: LayerData, 
            obs_names: pl.Series,
            feat_names: pl.Series):
        self.data = data
        self.obs_names = obs_names
        self.feat_names = feat_names
    
    
    # Methods ----
    def __str__(self):
        return f"Layer | dims: {self.dims}"
    
    def __repr__(self):
        return f"Layer | dims: {self.dims}"
    
    def assert_valid(self, assay_name = None, layer_name = None):
        match (assay_name, layer_name):
            case (None, None):
                e = "Unowned Layer : "
            case (None, str()):
                e = f"Unowned named Layer[{layer_name}] | "
            case (str(), str()):
                e = f"Assay[{assay_name}], Layer[{layer_name}] | "
            case _:
                e = f"Invalid (assay_name, layer_name) arguments: expected one of [({b('None'), b('None')}), ({b('None')},{b('str')}), ({b('str'), b('str')})] but got ({type(assay_name).__name__}, {type(layer_name).__name__})."
        
        # Check if the data is a dense numpy array or sparse scipy array
        if not isinstance(self.data, LayerData):
            e = e + f"Invalid data attribute type: expected one of {', '.join(e.__name__ for e in get_args(LayerData))}, but got {type(self.data).__name__}."
            
            logger.error(e)
            raise TypeError(e)
        
        # Check if the data array has 2 dimensions
        if len(self.data.shape) != 2:
            e = e + f"Invalid Layer dimension: expected shape (m,n) but got {self.data.shape}."
            
            logger.error(e)
            raise ValueError(e)
        
        # Check if obs_names length matches the number of rows in the data
        if self.obs_names.len() != self.data.shape[0]:
            e = e + f"Invalid number of observation names: obs_names has {self.obs_names.len()} elements but data attribute has {self.data.shape[0]} rows."
            
            logger.error(e)
            raise ValueError(e)
        
        # Check if feat_names length matches the number of columns in the data
        if self.feat_names.len() != self.data.shape[1]:
            e = f"Invalid number of feature names: feat_names has {self.feat_names.len()} elements but layer data has {self.data.shape[1]} features."
            
            logger.error(e)
            raise ValueError(e)
    
    
    # Properties ----
    @property
    def dims(self) -> tuple:
        # Check object is valid
        self.assert_valid()
        
        return self.data.shape


class LayerDict:
    layers: dict[str, Layer]
    owner: str | None
    
    def __init__(self, layers: dict[str, Layer], owner: str | None = None):
        self.layers = layers
        self.owner = owner
    
    def __str__(self) -> str:
        return str(self.layers)
    
    def __repr__(self) -> str:
        return repr(self.layers)
    
    def __getitem__(self, layer_name):
        match (layer_name in self.layers):
            case(True):
                return self.layers[layer_name]
            
            case(False): # Raise error
                e = f"Assay[{self.owner}] | Invalid layer name: layer {layer_name} does not exist in {list(self.layers.keys())}."
                
                logger.error(e)
                raise KeyError(e)
    
    def __setitem__(self, layer_name: str, layer: Layer):
        self.layers[layer_name] = layer
    
    def __delitem__(self, layer_name):
        match (layer_name in self.layers):
            case(True):
                del self.layers[layer_name]
            case(False):
                e = f"Assay[{self.owner}] | Invalid layer name: layer {layer_name} does not exist in {list(self.layers.keys())}."
                logger.error(e)
                
                raise KeyError(e)
    
    def __contains__(self, layer_name):
        return layer_name in self.layers
    
    def __iter__(self):
        return iter(self.layers)
    
    def __len__(self):
        return len(self.layers)
    
    def items(self):
        return self.layers.items()
    
    def keys(self):
        return self.layers.keys()
    
    def values(self):
        return self.layers.values()




class Assay:
    layers: LayerDict
    default_layer: str
    obs_md: pl.DataFrame
    feat_md: pl.DataFrame
    _obs_mship: pl.DataFrame
    _feat_mship: pl.DataFrame
    _obs_label: str
    _feat_label: str
    
    # Constructors ----
    @classmethod
    def from_layer(
            cls,
            layer: Layer,
            layer_name: str,
            assay_name : str | None = None,
            obs_md: None | pl.DataFrame = None,
            feat_md: None | pl.DataFrame = None) -> "Assay":
        """
        Construct an `Assay` object from data of a single layer.
        
        Parameters
        ----------
        `layer` : Layer
            The initial layer.
        `name` : str
            The name for the initial layer.
        `obs_names` : Union[str, Series]
            A string for a column in `obs_md` or a polars Series containing 
            the observation IDs.
        `feat_names` : Union[str, Series] 
            A string for a column in `feat_md` or a polars Series containing 
            the feature IDs.
        `obs_md` : Union[None, DataFrame]
            An optional metadata dataframe for observations.
        `feat_md` : Union[None, DataFrame]
            An optional metadata dataframe for features.
        
        Returns
        -------
        `Assay`
            An `Assay` object containing the data.
        """
        # Initialize an instance
        assay: Assay = cls()
        
        # Initialize layers dict
        assay.layers = LayerDict({})
        
        # Set default layer
        assay.default_layer = layer_name
        
        # Set labels
        assay._obs_label = layer.obs_names.name
        assay._feat_label = layer.feat_names.name
        
        # Handle `obs_md` options
        match (obs_md):
            case (pl.DataFrame()): # `obs_md` is a DataFrame
                obs_md = obs_md
                
                if assay._obs_label not in obs_md.columns:
                    w = f"{assay._obs_label} column was not found in observation metadata, will be included from the {layer_name} layer's observation names. Please include observation names in obs_md if this is not desired."
                    logger.warning(w)
                    
                    obs_md = obs_md.with_columns(layer.obs_names)
            
            case (None): # `obs_md` was not given
                obs_md = layer.obs_names.to_frame()
            
            case _: # `obs_md` is invalid
                e = f"obs_md argument should be one of [{b('DataFrame')}, {b('None')}] but got {b(type(obs_md).__name__)}."
                logger.error(e)
                
                raise TypeError(e)
        
        # Handle `feat_md` options
        match (feat_md):
            case (pl.DataFrame()): # `feat_md` is a DataFrame
                feat_md = feat_md
                
                if assay._feat_label not in feat_md.columns:
                    w = f"{assay._feat_label} column was not found in feature metadata, will be included from the {layer_name} layer's feature names. Please include feature IDs in feat_md if this is not desired."
                    logger.warning(w)
                    
                    feat_md = feat_md.with_columns(layer.feat_names)
            
            case (None): # `feat_md` was not given
                feat_md = layer.feat_names.to_frame()
            
            case _: # `feat_md` is invalid
                e = f"feat_md argument should be one of [{b('DataFrame')}, {b('None')}] but got {b(type(feat_md).__name__)}."
                logger.error(e)
                
                raise TypeError(e)
        
        # Set initial layer
        assay.layers[layer_name] = layer
        
        # Set metadata
        assay.obs_md = obs_md
        assay.feat_md = feat_md
        
        
        # Set layer membership
        assay._obs_mship = (
            layer.obs_names.to_frame()
            .with_columns(
                pl.col(assay._obs_label).is_in(layer.obs_names)
                .alias(layer_name)
            )
        )
        assay._feat_mship = (
            layer.feat_names.to_frame()
            .with_columns(
                pl.col(assay._feat_label).is_in(layer.feat_names)
                .alias(layer_name)
            )
        )
        
        
        return(assay)
    
    
    # Methods ----
    def assert_valid(self, assay_name: str | None = None):
        """
        Assert if an Assay is valid.
        
        Parameters
        ----------
        - `assay_name` (str) : The assay name to check.
        """
        # Prefix for potential error
        e = f"Assay[{assay_name}] | "
        
        # Check that all layers are present in membership dataframes
        if not set(self.layer_names).issubset(set(self._obs_mship.columns)):
            e = e + f"Layer names({self.layer_names}) do not match column names in observation membership dataframe({self._obs_mship.columns})."
            
            logger.error(e)
            raise ValueError(e)
        
        if not set(self.layer_names).issubset(set(self._feat_mship.columns)):
            e = f"Layer names({self.layer_names}) do not match column names in feature membership dataframe({self._feat_mship.columns})."
            
            logger.error(e)
            raise ValueError(e)
        
        # Iterate over assays
        for name, layer in self.layers.items():
            # Check if layer is valid       
            layer.assert_valid(assay_name, name)
            
            # Check layer dimensions match metadata.
            if not (
                layer.obs_names.len() <= self.obs_md.height and 
                layer.feat_names.len() <= self.feat_md.height
            ):
                e = e + f"Invalid layer dimension: expected shape to be smaller than ({self.obs_md.height}, {self.feat_md.height}) but got ({layer.data.shape})."
                
                logger.error(e)
                raise ValueError(e)
    
    
    @property
    def layer_names(self) -> list[str]:
        return list(self.layers.keys())


class AssayDict:
    assays: dict[str, Assay]
    
    def __init__(self, assays: dict[Assay]):
        self.assays = assays
    
    def __str__(self) -> str:
        return str(self.assays)
    
    def __repr__(self) -> str:
        return repr(self.assays)
    
    def __getitem__(self, assay_name):
        match (assay_name in self.assays):
            case(True):
                return self.assays[assay_name]
            case(False):
                e = f"Invalid assay name: assay {assay_name} does not exist in {list(self.assays.keys())}."
                logger.error(e)
                
                raise KeyError(e)
    
    def __setitem__(self, assay_name: str, assay: Assay):
        self.assays[assay_name] = assay
    
    def __delitem__(self, assay_name):
        match (assay_name in self.assays):
            case(True):
                del self.assays[assay_name]
            case(False):
                e = f"Invalid assay name: assay {assay_name} does not exist in {list(self.assays.keys())}."
                logger.error(e)
                
                raise KeyError(e)
    
    def __contains__(self, assay_name):
        return assay_name in self.assays
    
    def __iter__(self):
        return iter(self.assays)
    
    def __len__(self):
        return len(self.assays)
    
    def items(self):
        return self.assays.items()
    
    def keys(self):
        return self.assays.keys()
    
    def values(self):
        return self.assays.values()


class Experiment: 
    assays: AssayDict
    default_assay: str
    
    
    # Constructors ----
    @classmethod
    def from_layer(
            cls,
            layer: Layer,
            assay_name: str,
            layer_name: str,
            obs_md: None | pl.DataFrame = None,
            feat_md: None | pl.DataFrame = None) -> "Experiment":
        """
        Construct an `Experiment` object from a single layer.
        
        Parameters
        ----------
        - `layer` (Layer) : The initial layer.
        - `assay_name` (str): The name for the initial assay.
        - `layer_name` (str) : The name for the initial layer.
        - `obs_md` (None | DataFrame) : An optional metadata polars DataFrame 
            for observations.
        - `feat_md` (None | DataFrame) : An optional metadata polars DataFrame
            for features.
        
        Returns
        -------
        (Experiment) : An `Experiment` object containing the data.
        """
        # Initialize the instance
        experiment = cls()
        
        # Initialize assays
        experiment.assays = AssayDict({})
        
        # Set default assay
        experiment.def_assay = assay_name
        
        # Set assay
        experiment.assays[assay_name] = Assay.from_layer(layer, 
                                                         layer_name,
                                                         obs_md, 
                                                         feat_md)
        
        # Check object is valid
        experiment.assert_valid()
        
        return (experiment)
    
    
    # Methods ----
    def assert_valid(self):
        """
        Assert that an `Experiment` object is valid.
        """
        for name, assay in self.assays.items():
            # Assert that each assay is valid
            assay.assert_valid(name)
    
    def assay_names(self) -> list[str]:
        """
        Generate a list of current assays.
        
        Returns
        -------
        - (list[str]) : A list of assays.
        """
        return list(self.assays.keys())
    
    def layer_names(self, assay: str) -> list[str]:
        """
        Generate a list of current layers.
        
        Parameters
        ----------
        `assay` : `str`
            The named `Assay` to list existing layers for.
        
        Returns
        -------
        `list[str]`
            A list of layer names.
        """
        
        return list(self.assays[assay].layer_names)
    
    def get_layer(self, 
                 assay: str | None = None, 
                 layer: str | None = None) -> Layer:
        """
        Get a `Layer`.
        
        Parameters
        ----------
        
        `assay` : Union[str, None]
            The name of an assay. If not given uses default assay.
        
        `layer` : Union[str, None]
            The name of an assay. If not given uses default layer.
        
        Returns
        -------
        `Layer`
            A `Layer` object.
        """
        
        # Check object is valid
        self.assert_valid()
        
        return self.assays[assay].layers[layer]
    
    def set_layer(self, 
                  layer: Layer,
                  layer_name: str,
                  assay: str | None = None):
        # Handle options
        match (assay):
            case (None):
                assay = self.default_assay
            case (_):
                assay = assay
        
        # Handle whether assay already exists
        match (assay in self.assay_names()):
            case(True):
                self.assays[assay].layers[layer_name] = layer
            case(False):
                self.assays[assay] = Assay.from_layer(layer, layer_name)
        
        # Reference assay
        cur_assay: Assay = self.assays[assay]
        
        # Ensure observations and features are labelled correctly
        if (cur_assay._obs_label != layer.obs_names.name):
            w = f"Assay[{assay}], Layer[{layer_name}] | Mismatching observation labels: existing observations labelled '{cur_assay._obs_label}' but new layer has '{layer.obs_names.name}'. Overwriting observation label in new layer with '{cur_assay._obs_label}', are you sure this layer is part of the correct Assay?"
            logger.warning(w)
            
            layer.obs_names = layer.obs_names.rename(cur_assay._obs_label)
        if (cur_assay._feat_label != layer.feat_names.name):
            w = f"Assay[{assay}], Layer[{layer_name}] | Mismatching feature labels: existing features labelled '{cur_assay._feat_label}' but new layer has label '{layer.feat_names.name}'. Overwriting feature label in new layer with '{cur_assay._obs_label}', are you sure this layer is part of the correct Assay?"
            logger.warning(w)
            
            layer.feat_names = layer.feat_names.rename(cur_assay._feat_label)
        
        # Add layer membership
        cur_assay._obs_mship = cur_assay._obs_mship.with_columns(
            pl.col(cur_assay._obs_label)
            .is_in(layer.obs_names)
            .alias(layer_name)
        )
        cur_assay._feat_mship = cur_assay._feat_mship.with_columns(
            pl.col(cur_assay._feat_label)
            .is_in(layer.feat_names)
            .alias(layer_name)
        )
    
    
    def feat_md(self, assay: str | None = None) -> pl.DataFrame:
        """
        Access feature metadata for an assay.
        
        Parameters
        ----------
        `assay` : Union[str, None]
            The name of an assay. If None then uses default assay.
        
        Returns
        -------
        DataFrame 
            A polars DataFrame containing feature-level metadata.
        """
        
        # Check object is valid
        self.assert_valid()
        
        return self.assays[assay].feat_md
    
    def obs_md(self, assay: str | None = None) -> pl.DataFrame:
        """
        Access observation metadata for an assay.
        
        Parameters
        ----------
        - `assay` (str | None) : The name of an assay. If None then uses 
            default assay.
        
        Returns
        -------
        - (DataFrame) : A polars DataFrame containing observation-level 
            metadata.
        """
        
        # Check object is valid
        self.assert_valid()
        
        return self.assays[assay].obs_md

from turgnamh.core.experiment import Layer, Assay, Experiment

layer = Layer(np.ones(shape = (10,10)),
                  pl.Series(name = "10", values = np.arange(10)),
                  pl.Series(name = "hi", values = np.arange(10)))

data = Experiment.from_layer(
    layer = layer,
    assay_name = "RNA",
    layer_name = "counts"
)
import logging; logger.setLevel(logging.INFO)

data.set_layer(layer, "woohoo", "RNA")

data.set_layer(layer, "woohoo", "DNA")
