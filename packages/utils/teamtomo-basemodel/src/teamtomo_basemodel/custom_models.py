"""Abstract BaseModel class with import/export utils and schema generation."""

import json
import os
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict


class BaseModelTeamTomo(BaseModel):
    """Implementation of a Pydantic BaseModel with additional, useful utilities.

    This class currently only provides import/export functions for both JSON and YAML
    files as well as a schema generation alias for the `model_json_schema` method.
    Definition of the fields and their validation is the responsibility of subclasses.

    The model is configured to:
    1. Forbid extra fields not defined in the model. This helps catch errors early.
    2. Allow arbitrary types, like tensors or DataFrames, to also hold data.

    Attributes
    ----------
    model_config : ConfigDict
        Configuration for the Pydantic model

    Methods
    -------
    from_json(json_path: str | os.PathLike) -> "BaseModelTeamTomo":
        Load a BaseModelTeamTomo from a serialized JSON file.
    from_yaml(yaml_path: str | os.PathLike) -> "BaseModelTeamTomo":
        Load a BaseModelTeamTomo from a serialized YAML file.
    to_json(json_path: str | os.PathLike, **kwargs) -> None:
        Serialize the BaseModelTeamTomo to a JSON file.
    to_yaml(yaml_path: str | os.PathLike, **kwargs) -> None:
        Serialize the BaseModelTeamTomo to a YAML file.
    get_schema(**kwargs) -> dict[str, Any]:
        Alias for `model_json_schema` method to get the JSON schema of the model.
    """

    model_config: ConfigDict = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    #####################################
    ### Import/instantiation methods ###
    #####################################

    @classmethod
    def from_json(cls, json_path: str | os.PathLike) -> "BaseModelTeamTomo":
        """Load a BaseModelTeamTomo from a serialized JSON file.

        Parameters
        ----------
        json_path : str | os.PathLike
            Path to the JSON file to load.

        Returns
        -------
        BaseModelTeamTomo
            Instance of the BaseModelTeamTomo subclass loaded from the JSON file.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str | os.PathLike) -> "BaseModelTeamTomo":
        """Load a BaseModelTeamTomo from a serialized YAML file.

        Parameters
        ----------
        yaml_path : str | os.PathLike
            Path to the YAML file to load.

        Returns
        -------
        BaseModelTeamTomo
            Instance of the BaseModelTeamTomo subclass loaded from the YAML file.
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    ####################################
    ### Export/serialization methods ###
    ####################################

    def to_json(self, json_path: str | os.PathLike, **kwargs: Any) -> None:
        """Serialize the BaseModelTeamTomo to a JSON file.

        Parameters
        ----------
        json_path : str | os.PathLike
            Path to the JSON file to save.
        **kwargs : dict
            Additional keyword arguments to pass to `model_dump` method.

        Returns
        -------
        None
        """
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(**kwargs), f)

    def to_yaml(self, yaml_path: str | os.PathLike, **kwargs: Any) -> None:
        """Serialize the BaseModelTeamTomo to a YAML file.

        Parameters
        ----------
        yaml_path : str | os.PathLike
            Path to the YAML file to save.
        **kwargs : dict
            Additional keyword arguments to pass to `model_dump` method.

        Returns
        -------
        None
        """
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(**kwargs), f)

    #################################
    ### Schema generation methods ###
    #################################

    def get_schema(self, **kwargs: Any) -> dict[str, Any]:
        """Alias for `model_json_schema` method to get the JSON schema of the model."""
        return self.model_json_schema(**kwargs)  # type: ignore[no-any-return]
