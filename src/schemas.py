import os
import pandas as pd

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

## Inferencing
class Statement(BaseModel):
    label: str = Field(..., title="Label", description="The label of the statement")
    text: str = Field(..., title="Text", description="The text of the statement")
    

class IssueOwner(BaseModel):
    name: str = Field(..., title="Full Name", description="The full name of the issue owner")
    short_name: Optional[str] = Field(None, title="Short Name", description="The short name of the issue owner")
    name_mapped: Optional[str] = Field(None, title="Name Mapped", description="The name mapped to the issue owner")
    
    def __key(self) -> tuple:
        return (self.name_mapped)
    
    def __eq__(self, value) -> bool:
        return super().__eq__(value)
    
    def __hash__(self) -> int:
        return hash(self.__key())
    
    def get_name_mapped(self, name_mapping: dict):
        
        # check if self.name in mapping key or value
        for key, variant_list in name_mapping.items():
            if self.name == key:
                self.name_mapped = key
                continue
            for name in variant_list:
                if self.name == name:
                    self.name_mapped = key
        
        if self.name_mapped:
            return self.name_mapped
        
        raise ValueError(f"Name {self.name} not found in mapping.")
    
    
class IssueItem(BaseModel):
    election_id: str = Field(..., title="Election ID", description="The ID of the election (format: 'YYYY/election')")
    statement: Statement = Field(..., title="Statement", description="The statement")
    issue_owner: IssueOwner = Field(..., title="Issue Owner", description="The issue owner")
    answer: Optional[str] = Field(..., title="Answer", description="The answer: 'Stimme zu', 'Neutral', 'Stimme nicht zu'")
    comment: str = Field(None, title="Comment", description="Optional comment from the party")
    prompt: str = Field(..., title="Prompt", description="The prompt")
    prompt_variant_idx: int = Field(..., title="Prompt Variant Index", description="The index of the prompt variant")
    

class InferenceItem(IssueItem):
    model_id: str = Field(..., title="Model ID", description="The model's unique identifier")
    llm_response: str = Field(..., title="LLM Response", description="The response of the LLM")
    path_activation_dir: str = Field(..., title="Activation Directory", description="The directory of the activations")
    
    @property
    def path_activation_file(self) -> str:
        # create unique path for the activation file
        file_path = f"{self.election_id.replace('/','-')}_{self.issue_owner.name.replace(' ','-')}_{self.statement.label.replace('.','')}_{self.prompt_variant_idx}.pt"
        
        path = os.path.join(self.path_model_activation_dir, file_path)
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return path
    
    @property
    def path_logits_file(self) -> str:
        # create unique path for the logits file
        file_path = f"{self.election_id.replace('/','-')}_{self.issue_owner.name.replace(' ','-')}_{self.statement.label.replace('.','')}_{self.prompt_variant_idx}_logits.pt"
        
        path = os.path.join(self.path_model_activation_dir, file_path)
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return path
    
    @property
    def path_model_activation_dir(self) -> str:
        path = os.path.join(self.path_activation_dir, self.model_id.replace("/","-"))
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        return path
    

class Issues(BaseModel):
    issue_items: list[IssueItem] = Field(..., title="Issues", description="A list of issues")
    data_dir: str = Field(..., title="Data Directory", description="The directory of the issues")
    issue_config: dict = Field(..., title="Issue Config", description="The configuration of the dataset")
    country: str = Field(..., title="Country Name", description="The name of the country")
        
    @property
    def date(self) -> datetime:
        return datetime.now()
    
    @property
    def path(self) -> str:
        # Return the path to the issues composed of date and length of data
        file_path = f"{self.date.strftime('%Y-%m-%d_%H-%M-%S')}_issues_{len(self.issue_items)}.json"
        path = os.path.join(self.data_dir, self.country.lower(), file_path)
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        return path

    def __len__(self):
        return len(self.issue_items)
    
    def __getitem__(self, idx):
        return self.issue_items[idx].model_dump()
    
    def save_as_json(self):
        json = self.model_dump_json(indent=4)
        
        with open(self.path, 'w') as f:
            f.write(json)
    

class Dataset(BaseModel):
    data: list[InferenceItem] = Field(..., title="Dataset", description="A list of inference data entries")
    data_dir: str = Field(..., title="Data Directory", description="The directory of the dataset")
    issue_config: dict = Field(..., title="Issue Config", description="The configuration of the dataset")
    inference_config: dict = Field(..., title="Inference Config", description="The configuration of the inference")
    country: str = Field(..., title="Country Name", description="The name of the country")
        
    @property
    def date(self) -> datetime:
        return datetime.now()
    
    @property
    def path(self) -> str:
        # Return the path to the dataset composed of date and length of data
        file_path = f"{self.date.strftime('%Y-%m-%d_%H-%M-%S')}_dataset_{len(self.data)}.json"
        path = os.path.join(
            self.data_dir,
            self.country.lower(),
            self.model_ids[0],
            file_path
        )
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        return path
    
    @property
    def issue_owner_names(self) -> list[str]:
        
        # Use a set to collect unique issue owner names
        unique_names = list()
        for item in self.data:
            name_mapping = self.issue_config['name_mapping']
            unique_name = item.issue_owner.get_name_mapped(name_mapping=name_mapping)
            unique_names.append(unique_name)
        
        unique_names = {item for item in unique_names}
        
        # Return as a list
        return list(unique_names)
    
    @property
    def issue_owner_short_names(self) -> list[str]:
        # Use a set to collect unique issue owner names
        unique_names = {item.issue_owner.short_name for item in self.data}
        
        # Return as a list
        return list(unique_names)
    
    @property
    def model_ids(self) -> list[str]:
        # Use a set to collect unique model ids
        unique_ids = {item.model_id for item in self.data}
        
        # Return as a list
        return list(unique_ids)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].model_dump()
    
    def save_as_json(self):
        json_file = self.model_dump_json(indent=4)
        
        with open(self.path, 'w') as f:
            f.write(json_file)
            
    def get_all_unique_issue_owners(self, model_id=None) -> list[dict[str, str]]:
        
        unique_ios = set()
        mapped_names = set()
        
        for item in self.data:
            
            if item.issue_owner.name_mapped is None:
                name_mapping = self.issue_config['name_mapping']
                _ = item.issue_owner.get_name_mapped(name_mapping=name_mapping)
                
            if model_id and item.model_id != model_id:
                continue
            
            if item.issue_owner.name_mapped not in mapped_names:
                unique_ios.add(item.issue_owner)
                mapped_names.add(item.issue_owner.name_mapped)
        
        return unique_ios


## Probing
class Metrics(BaseModel):
    accuracy: float = Field(..., title="Accuracy", description="The accuracy of the model", ge=0, le=1)
    f1: float = Field(..., title="F1 Score", description="The F1 score of the model", ge=0, le=1)
    precision: float = Field(..., title="Precision", description="The precision of the model", ge=0, le=1)
    recall: float = Field(..., title="Recall", description="The recall of the model", ge=0, le=1)


class ValueProbe(BaseModel):
    model_id: str = Field(..., title="Model ID", description="The model's unique identifier")
    issue_owner: IssueOwner = Field(..., title="Issue Owner", description="The issue owner")
    metrics: Metrics = Field(..., title="Metrics", description="Performance metrics for the model")
    layer_min: int = Field(..., title="Layer Min", description="The minimum layer number used for probing", ge=0)
    layer_max: int = Field(..., title="Layer Max", description="The maximum layer number used for probing", ge=0)
    path_value_probe_dir: str = Field(..., title="Value Probe Directory", description="The directory of the value probes")
    top_tokens: list[str] | None = Field(None, title="Top Tokens", description="Top tokens of the value probe")
    top_value_vector_tokens: list[str] | None = Field(None, title="Top Value Vector Tokens", description="Top tokens of the value vector")
    similarity: dict[str, float] = Field(
        {},
        title="Similarity",
        description="The similarity of the value probe to other value probes with format '{<model_id>_<issue_owner.name>: 'similarity'}'"
    )
    country: str = Field(..., title="Country Name", description="The name of the country")
    
    @property
    def date(self) -> datetime:
        return datetime.now()
    
    @property
    def path_value_probe(self) -> str:
        # create unique path for the value probe
        file_path = f"{self.issue_owner.name_mapped.replace(' ','-')}_value_probe_{self.layer_min}-{self.layer_max}.pt"
        
        path = os.path.join(
            self.path_value_probe_model_dir,
            file_path
        )
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return path
    
    @property
    def path_value_probe_model_dir(self) -> str:
        
        path = os.path.join(
            self.path_value_probe_dir,
            self.country.lower(),
            self.model_id,
        )
        
        return path

    def __key(self) -> tuple:
        return (self.issue_owner.name_mapped, self.model_id, self.layer_min, self.layer_max)
    
    def __eq__(self, value) -> bool:
        return super().__eq__(value)
    
    def __hash__(self) -> int:
        return hash(self.__key())

## Vector Extraction
class ValueVector(BaseModel):
    """
    A class that represents a value vector extracted from a model.
    It is used to store the value vectors for the probing results.
    """
    value_probe: ValueProbe = Field(..., title="Value Probe", description="The value probe")
    cos_similarity: float = Field(..., title="Cosine Similarity", description="The cosine similarity of the value vector with the value probe")
    layer: int = Field(..., title="Layer", description="The layer number in the neural network", ge=0)
    mlp: int = Field(..., title="MLP", description="The MLP (Multi-layer Perceptron) index", ge=0)
    path_value_vector_dir: str = Field(..., title="Value Probe Directory", description="The directory of the value probes")
    
    @property
    def path_value_vector(self) -> str:
        # create unique path for the value vector
        file_path = f"value_vector_layer-{self.layer}_mlp-{self.mlp}.pt"
        file_path = file_path.replace(' ','-').replace('/','-')
        
        path = os.path.join(
            self.path_value_vector_dir,
            self.value_probe.country.lower(),
            self.value_probe.model_id,
            self.value_probe.issue_owner.name_mapped,
            file_path
        )
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return path


class DiametricValueVector(ValueVector):
    """
    A subclass of ValueVector that represents a diametric (negative) value vector.
    It is used to store the diametric value vectors for the probing results.
    """
    
    @property
    def path_value_vector(self) -> str:
        # create unique path for the diametric value vector
        file_path = f"diametric_value_vector_layer-{self.layer}_mlp-{self.mlp}.pt"
        file_path = file_path.replace(' ','-').replace('/','-')
        
        path = os.path.join(
            self.path_value_vector_dir,
            self.value_probe.country.lower(),
            self.value_probe.model_id,
            self.value_probe.issue_owner.name_mapped,
            file_path
        )
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return path

  
class Probing(BaseModel):
    value_probes: list[ValueProbe] = Field(..., title="Value Probes", description="list of probing results")
    path_probing_dir: str = Field(..., title="Value Probe Directory", description="The directory of the value probes")
    probing_config: dict = Field(..., title="Config", description="The configuration of the probing results")
    value_vectors: Optional[list[ValueVector | DiametricValueVector]] = Field(None, title="Value Vectors", description="list of value vectors")
    country: str = Field(..., title="Country Name", description="The name of the country")
    
    @property
    def date(self) -> datetime:
        return datetime.now()
    
    @property
    def path(self) -> str:
        # Return the path to the probing results composed of date and length of value probes
        if not self.value_vectors:
            file_path = f"{self.date.strftime('%Y-%m-%d_%H-%M-%S')}_probing_{len(self.value_probes)}.json"
        if self.value_vectors:
            file_path = f"{self.date.strftime('%Y-%m-%d_%H-%M-%S')}_probing_{len(self.value_probes)}_value_vectors_{len(self.value_vectors)}.json"
        path = os.path.join(
            self.path_probing_dir,
            self.country.lower(),
            self.model_ids[0],
            file_path,
        )
        
        # Check if the path and directory exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        return path
    
    @property
    def model_ids(self) -> list[str]:
        # Use a set to collect unique model ids
        unique_ids = {item.model_id for item in self.value_probes}
        
        # Return as a list
        return list(unique_ids)

    def __len__(self):
        return len(self.value_probes)
    
    def __getitem__(self, idx):
        return self.value_probes[idx].model_dump()
    
    def save_as_json(self):
        json = self.model_dump_json(indent=4)
        
        with open(self.path, 'w') as f:
            f.write(json)
    
    def get_all_value_probes(self, model_id=None) -> list[dict[str, str]]:
        
        if model_id:
            return [item for item in self.value_probes if item.model_id == model_id]
        else:
            return [item for item in self.value_probes]

    def get_all_value_vectors(self, model_id=None, issue_owner_name_mapped=None) -> list[dict[str, str]]:
        
        value_vectors = [item for item in self.value_vectors]
        
        if model_id:
            value_vectors = [item for item in value_vectors if item.value_probe.model_id == model_id]
        if issue_owner_name_mapped:
            value_vectors = [item for item in value_vectors if item.value_probe.issue_owner.name_mapped == issue_owner_name_mapped]

        return value_vectors
    
    def get_all_issuer_owner_names_mapped(self, model_id=None) -> list[str]:
        
        value_probes = self.get_all_value_probes(model_id=model_id)
        
        # Use a set to collect unique issue owner names
        unique_names = {item.issue_owner.name_mapped for item in value_probes}
        
        return list(unique_names)


class Persona(BaseModel):
    
    age: str = Field(..., title="Age", description="The age of the persona")
    gender: str = Field(..., title="Gender", description="The gender of the persona")
    education: str = Field(..., title="Education", description="The education of the persona")
    employment: str = Field(..., title="Employment", description="The employment of the persona")
    hhincome: str = Field(..., title="Household Income", description="The household income of the persona")
    left_leaning: str = Field(..., title="Left Leaning", description="The persona is left leaning")
    immigration: str = Field(..., title="Immigration", description="The persona is immigrated")
    inequality: str = Field(..., title="Inequality", description="The persona is affected by inequality")
    year_of_election: str = Field(..., title="Year of Election", description="The year of the election")
            
    def get_prompt_variants(self, prompt_variant_template: str) -> str:
        """
        Generate prompt variants by filling in the persona attributes.
        
        Args:
            prompt_variant_template (str): The template for the prompt variant
            
        Returns:
            str: The prompt variant with filled in persona attributes
        """

        # Get only non-None attributes
        persona_config = {k: v for k, v in self.model_dump().items() if v is not None}

        # Format the cleaned template with available attributes
        return prompt_variant_template.format(**persona_config)


class KeyValue(BaseModel):
    value_vector: ValueVector | DiametricValueVector = Field(..., title="Value Vector", description="The value vector")
    persona: Persona = Field(..., title="Persona", description="The persona")
    value: float = Field(..., title="Activation Value", description="The activation value contributed by the key value to the value vector")
    scale: float = Field(..., title="Scale", description="The scaling factor applied to the value and calculated by the cosine similarity of the value vector")
    mean_probability: float = Field(..., title="Mean Probability", description="The mean probability of the model's response")
    first_probability: float = Field(..., title="First Token Probability", description="The probability of the first token in the model's response")
    first_3_probability: float = Field(..., title="First 3 Tokens Probability", description="The mean probability of the first 3 tokens in the model's response")
    persona_prompt_variant_idx: float = Field(..., title="Prompt Variant Index", description="The index of the prompt variant")
    response: str = Field(..., title="Response", description="The response of the model")
    path_key_value_dir: str = Field(..., title="Key Value Directory", description="The directory of the key values")

    def to_flat_dict(self) -> dict[str, str]:
        """
        Flatten this KeyValue instance (and its nested attributes)
        into a dictionary for CSV export.
        
        The returned dictionary includes:
          - KeyValue-level attributes (normed_value_mean, value_std, path_key_value_dir, ...)
          - ValueVector-level attributes (layer, mlp, path_value_vector_dir, ...)
          - ValueProbe-level attributes inside ValueVector.value_probe
            (model_id, layer, path_value_probe_dir, plus the subfields in metrics, etc.)
          - IssueOwner attributes (name, short_name, name_mapped)
          - Persona attributes (age, gender, education, ...)
        
        Excludes any actual PyTorch tensor data itself.
        """
        # --- 1) Pull top-level KeyValue fields ---
        row = {
            "key_value.value": self.value,
            "key_value.scale": self.scale,
            "key_value.mean_probability": self.mean_probability,
            "key_value.first_probability": self.first_probability,
            "key_value.first_3_probability": self.first_3_probability,
            "key_value.persona_prompt_variant_idx": self.persona_prompt_variant_idx,
            "key_value.response": self.response,
            "key_value.path_key_value_dir": self.path_key_value_dir,
        }
        
        # --- 2) Flatten ValueVector info ---
        vv = self.value_vector
        row.update({
            "value_vector.layer": vv.layer,
            "value_vector.mlp": vv.mlp,
            "value_vector.path_value_vector_dir": vv.path_value_vector_dir,
        })
        
        # --- 3) Flatten ValueProbe info within ValueVector ---
        vp = vv.value_probe
        row.update({
            "value_probe.model_id": vp.model_id,
            "value_probe.layer_min": vp.layer_min,
            "value_probe.layer_max": vp.layer_max,
            "value_probe.path_value_probe_dir": vp.path_value_probe_dir,
        })
        
        # 3a) Flatten IssueOwner inside ValueProbe
        owner = vp.issue_owner
        row.update({
            "issue_owner.name": owner.name,
            "issue_owner.short_name": owner.short_name,
            "issue_owner.name_mapped": owner.name_mapped,
        })
        
        # 3b) Flatten Metrics inside ValueProbe
        metrics = vp.metrics
        row.update({
            "metrics.accuracy": metrics.accuracy,
            "metrics.f1": metrics.f1,
            "metrics.precision": metrics.precision,
            "metrics.recall": metrics.recall,
        })
        
        # --- 4) Flatten Persona info ---
        persona = self.persona
        
        row.update({
            "persona.age": persona.age,
            "persona.gender": persona.gender,
            "persona.education": persona.education,
            "persona.employment": persona.employment,
            "persona.hhincome": persona.hhincome,
            "persona.left_leaning": persona.left_leaning,
            "persona.immigration": persona.immigration,
            "persona.inequality": persona.inequality,
            "persona.year_of_election": persona.year_of_election,
        })
        
        return row

    @classmethod
    def to_dataframe(self, key_values: list["KeyValue"]) -> pd.DataFrame:
        """
        Convert a list of KeyValue objects into a Pandas DataFrame
        with one row per KeyValue. Each column is a flattened attribute.
        """
        rows = [kv.to_flat_dict() for kv in key_values]
        df = pd.DataFrame(rows)
        return df

    @classmethod
    def save_to_csv(self, key_values: list["KeyValue"], csv_path: str) -> None:
        """
        Flatten a list of KeyValue objects into a DataFrame and save to CSV.
        """
        df = self.to_dataframe(key_values)
        df.to_csv(csv_path, index=False)
