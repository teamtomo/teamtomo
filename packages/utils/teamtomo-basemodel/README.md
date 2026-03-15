# teamtomo-basemodel
Custom Pydantic classes and type-hints for building more complex data classes within TeamTomo.

## Package Usage

The `BaseModelTeamTomo` should be used when making sub-classes which should inherit the basic parsing/serialization functionality within Pydantic. Currently, the following methods are implemented:

- `to_json`/`to_yaml` - Export the model fields to a JSON/YAML file (filepath passed as argument).
- `from_json`/`from_yaml` - Instantiate a new model from contents in a JSON/YAML file (filepath passed as argument).
- `get_schema` - Alias for `model_json_schema` method of the Pydantic basemodel class.

### Basic example

These derived classes define their own fields and validation methods. For example, the following is an example derived class.

```python
from pydantic import Field

from teamtomo_basemodel import BaseModelTeamTomo


class CustomModel(BaseModelTeamTomo):
    """An example custom model with three fields."""

    some_label: str
    some_value: float = Field(ge=0.0)  # will raise Validation error if less than 0.0
    list_int: list[int] = Field(description="A list of integers")


my_model = CustomModel(some_label="Example", some_value=42.0, list_int=[1, 2, 3, 4, 5])

# Export the model to a YAML file
my_model.to_yaml("custom_model.yaml")

# Create new model from the YAML file
new_model = CustomModel.from_yaml("custom_model.yaml")

# Check if the two models are equal
print(my_model == new_model)  # prints 'True'
```

Inspecting the contents of the `custom_model.yaml` file, we see the following.

```yaml
list_int:
- 1
- 2
- 3
- 4
- 5
some_label: Example
some_value: 42.0
```

### Obtaining model schema

The JSON model schema for a Pydantic model is used to define and validate the structure, its data types, and other constraints of the represented data. This ensures consistency across data types, and we expose the `get_schema` method for this purpose.

```Python
from pprint import pprint

# Using the same 'CustomModel' class defined above
my_schema = my_model.get_schema()

print(type(my_schema))
# <class 'dict'>

pprint(my_schema)
# {'additionalProperties': False,
#  'description': 'An example custom model with three fields.',
#  'properties': {'list_int': {'description': 'A list of integers',
#                              'items': {'type': 'integer'},
#                              'title': 'List Int',
#                              'type': 'array'},
#                 'some_label': {'title': 'Some Label', 'type': 'string'},
#                 'some_value': {'minimum': 0.0,
#                                'title': 'Some Value',
#                                'type': 'number'}},
#  'required': ['some_label', 'some_value', 'list_int'],
#  'title': 'CustomModel',
#  'type': 'object'}
```

### Using included type-hints

Some data structures used by custom data classes may be serialized elsewhere (e.g., DataFrames using csv files) and should not be serialized by the class. For this reason, we also include helpful type-hints which tell Pydantic to ignore these fields while still exposing them to the user. The currently included type-hints are:

- `ExcludedTensor`
- `ExcludedNumpyArray`
- `ExcludedDataFrame`

As an example, we can define another model which uses the field `df_path` to load in a DataFrame; the `df_path` field is serialized, but the DataFrame itself is not.

```python
import pandas as pd
from teamtomo_basemodel import BaseModelTeamTomo, ExcludedDataFrame


class CustomModelWithDF(BaseModelTeamTomo):
    """An example custom model with a DataFrame field."""

    df_path: str
    df: ExcludedDataFrame
    
    def __init__(self, **data):
        super().__init__(**data)
        self.df = pd.read_csv(self.df_path) # Load the DataFrame from the specified path


# First, write an example DataFrame to a CSV file
df_example = pd.DataFrame({"column1": [1, 2, 3], "column2": ["a", "b", "c"]})
df_example.to_csv("example_df.csv", index=False)

# Now, create an instance of the custom model with the DataFrame
custom_model_with_df = CustomModelWithDF(df_path="example_df.csv")
print(custom_model_with_df.df)
#    column1 column2
# 0        1       a
# 1        2       b
# 2        3       c

# Print the contents of the model
print(custom_model_with_df.model_dump())
# {'df_path': 'example_df.csv'}
```

## Installation

TeamTomo basemodel is available on the Python package index and can be installed via

```bash
pip install teamtomo-basemodel
```

### From source

If you'd rather install the package from source, say for development purposes, use the following commands for cloning then installing a local editable version of the package

```bash
git clone https://github.com/teamtomo/teamtomo-basemodel.git
cd teamtomo-basemodel
pip install -e .
```


