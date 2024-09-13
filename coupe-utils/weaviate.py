# weaviate_utils.py

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from weaviate.classes.query import HybridFusion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeaviateUtils:
    def __init__(self, wcs_url: str, wcs_key: str, openai_api_key: str):
        self.wcs_url = wcs_url
        self.wcs_key = wcs_key
        self.openai_api_key = openai_api_key
        self.client = self.initiate_weaviate_client()
        self._schema_properties = []

    def initiate_weaviate_client(
        self, additional_headers: Optional[Dict[str, str]] = None
    ) -> weaviate.Client:
        headers = {"X-OpenAI-Api-key": self.openai_api_key}
        if additional_headers:
            headers.update(additional_headers)
        client = weaviate.connect_to_wcs(
            cluster_url=self.wcs_url,
            auth_credentials=weaviate.auth.AuthApiKey(self.wcs_key),
            headers=headers,
        )
        return client

    @property
    def schema_properties(self) -> List[Dict[str, Any]]:
        return self._schema_properties

    @schema_properties.setter
    def schema_properties(self, properties: List[Dict[str, Any]]) -> None:
        self._schema_properties = properties

    def generate_weaviate_properties(self) -> List[Property]:
        return [
            Property(
                name=prop["name"],
                data_type=DataType(prop["data_type"]),
                nested_properties=[
                    Property(
                        name=nested_prop["name"],
                        data_type=DataType(nested_prop["data_type"]),
                    )
                    for nested_prop in prop.get("nested_properties", [])
                ],
            )
            for prop in self.schema_properties
        ]

    def _generate_property_from_data_structure(
        self, data_structure: Union[Dict, List], name: str
    ) -> Dict:
        """Generates a property definition based on a data structure."""
        if isinstance(data_structure, dict):
            data_type = DataType.OBJECT
            nested_properties = [
                self._generate_property_from_data_structure(value, nested_name)
                for nested_name, value in data_structure.items()
            ]
            return {
                "name": name,
                "data_type": data_type,
                "nested_properties": nested_properties,
            }
        elif isinstance(data_structure, list):
            if data_structure and isinstance(data_structure[0], dict):
                data_type = DataType.OBJECT_ARRAY
                nested_properties = [
                    self._generate_property_from_data_structure(value, nested_name)
                    for nested_name, value in data_structure[0].items()
                ]
                return {
                    "name": name,
                    "data_type": data_type,
                    "nested_properties": nested_properties,
                }
            else:
                data_type = DataType.TEXT_ARRAY
                return {"name": name, "data_type": data_type}
        elif isinstance(data_structure, str):
            return {"name": name, "data_type": DataType.TEXT}
        elif isinstance(data_structure, bool):
            return {"name": name, "data_type": DataType.BOOL}
        elif isinstance(data_structure, int):
            return {"name": name, "data_type": DataType.INT}
        elif isinstance(data_structure, float):
            return {"name": name, "data_type": DataType.NUMBER}
        elif isinstance(data_structure, datetime):
            return {"name": name, "data_type": DataType.DATE}
        else:
            raise TypeError(
                f"Unsupported data type: {type(data_structure)} for property '{name}'"
            )

    def generate_schema_from_data(self, data: Union[Dict, List]) -> None:
        """Generates schema properties from a sample data object."""
        if isinstance(data, dict):
            self._schema_properties = [
                self._generate_property_from_data_structure(value, name)
                for name, value in data.items()
            ]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            self._schema_properties = [
                self._generate_property_from_data_structure(value, name)
                for name, value in data[0].items()
            ]
        else:
            raise TypeError(
                "Input data must be a dictionary or a non-empty list of dictionaries."
            )

    def create_collection(
        self,
        collection_name: str,
        vectorizer_config: Configure.Vectorizer = Configure.Vectorizer.text2vec_openai(
            model="text-embedding-ada-002"
        ),
        reranker_config: Optional[Configure.Reranker] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> weaviate.Collection:
        if reranker_config is None and additional_headers is None:
            self.client = self.initiate_weaviate_client()
        else:
            self.client = self.initiate_weaviate_client(additional_headers)

        weaviate_properties = self.generate_weaviate_properties()

        self.client.collections.create(
            collection_name,
            vectorizer_config=vectorizer_config,
            reranker_config=reranker_config,
            properties=weaviate_properties,
        )
        return self.client.collections.get(collection_name)

    def process_dates(self, data_row: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in data_row.items():
            if isinstance(value, datetime):
                data_row[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        data_row[key][i] = self.process_dates(item)
        return data_row

    def upload_to_weaviate(
        self, collection_name: str, data: List[Dict[str, Any]]
    ) -> None:
        collection = self.client.collections.get(collection_name)
        with collection.batch.fixed_size(batch_size=100) as batch:
            for data_row in data:
                data_row = self.process_dates(data_row)
                batch.add_object(properties=data_row, uuid=data_row.get("uuid"))

    def add_single_object_to_weaviate(
        self, collection_name: str, data_row: Dict[str, Any]
    ) -> None:
        collection = self.client.collections.get(collection_name)
        try:
            data_row = self.process_dates(data_row)
            collection.data.insert(data_row, uuid=data_row.get("uuid"))
        except weaviate.exceptions.UnexpectedStatusCodeError as e:
            if "already exists" in str(e):
                logger.warning(
                    f"Object with ID {data_row.get('uuid')} already exists in Weaviate. Skipping."
                )
            else:
                raise

    def simple_query(
        self,
        collection_name: str,
        query: str,
        limit: int = 100,
        fusion_type: HybridFusion = HybridFusion.RANKED,
        auto_limit: int = 4,
        alpha: float = 0.4,
    ) -> List[Dict[str, Any]]:
        collection = self.client.collections.get(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' not found.")

        results = []
        try:
            response = collection.query.hybrid(
                query=query,
                limit=limit,
                fusion_type=fusion_type,
                auto_limit=auto_limit,
                alpha=alpha,
            )
            for o in response.objects:
                results.append(
                    {
                        key: self.process_dates(value)
                        for key, value in o.properties.items()
                    }
                )
        except Exception as e:
            raise
        return results
