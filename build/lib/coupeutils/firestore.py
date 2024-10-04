from google.cloud import firestore
from google.api_core import exceptions
from typing import Dict, List, Any, Optional, Tuple


class FirestoreUtils:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.db = self.initialize_firestore_client()

    def initialize_firestore_client(self) -> firestore.Client:
        try:
            db = firestore.Client(project=self.project_id)
            return db
        except Exception as e:
            raise Exception(f"Error initializing Firestore client: {str(e)}")

    def get_document(
        self, document_id: str, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a document from Firestore."""
        try:
            doc_ref = self.db.collection(collection_name).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            else:
                print(
                    f"Document {document_id} not found in collection {collection_name}"
                )
                return None
        except Exception as e:
            raise Exception(f"Error retrieving document {document_id}: {str(e)}")

    def bulk_store_documents(
        self, data_list: List[Dict[str, Any]], collection_name: str
    ) -> None:
        """Adds multiple documents to a Firestore collection using batch operations."""
        try:
            batch = self.db.batch()
            for data in data_list:
                doc_ref = self.db.collection(collection_name).document(
                    data.get("document_id", None)
                )
                batch.set(doc_ref, data)
            batch.commit()
        except Exception as e:
            raise Exception(
                f"Error during bulk store to collection {collection_name}: {str(e)}"
            )

    def save_document(
        self,
        data: Dict[str, Any],
        document_id: Optional[str] = None,
        collection_name: str = None,
    ) -> None:
        """Saves a document to a Firestore collection."""
        if collection_name is None:
            raise ValueError("collection_name cannot be None")
        try:
            doc_ref = (
                self.db.collection(collection_name).document(document_id)
                if document_id
                else self.db.collection(collection_name).document()
            )
            doc_ref.set(data)
        except Exception as e:
            raise Exception(
                f"Error saving document to collection {collection_name}: {str(e)}"
            )

    def query_collection(
        self, query_params: Dict[str, Tuple[str, Any]], collection_name: str
    ) -> List[Dict[str, Any]]:
        """Queries a Firestore collection based on specified parameters and operators."""
        try:
            query = self.db.collection(collection_name)
            for field, (operator, value) in query_params.items():
                if operator not in (
                    "==",
                    ">",
                    ">=",
                    "<",
                    "<=",
                    "array_contains",
                    "in",
                    "array_contains_any",
                    "not_in",
                ):
                    raise ValueError(f"Unsupported operator: {operator}")
                query = query.where(field, operator, value)
            return [doc.to_dict() for doc in query.stream()]
        except Exception as e:
            raise Exception(
                f"Error during query execution on collection {collection_name}: {str(e)}"
            )

    def update_document(
        self, document_id: str, update_data: Dict[str, Any], collection_name: str
    ) -> None:
        """Updates a single document in Firestore."""
        try:
            doc_ref = self.db.collection(collection_name).document(document_id)
            doc_ref.update(update_data)
        except exceptions.InvalidArgument as e:
            raise Exception(
                f"Error updating document {document_id}: Invalid argument - {str(e)}"
            )
        except Exception as e:
            raise Exception(f"Error updating document {document_id}: {str(e)}")

    def bulk_update_documents(
        self, update_data_list: List[Tuple[str, Dict[str, Any]]], collection_name: str
    ) -> None:
        """Updates multiple documents in Firestore using batch operations."""
        try:
            batch = self.db.batch()
            for object_id, update_data in update_data_list:
                doc_ref = self.db.collection(collection_name).document(object_id)
                batch.update(doc_ref, update_data)
            batch.commit()
        except Exception as e:
            raise Exception(
                f"Error during bulk update in collection {collection_name}: {str(e)}"
            )

    def get_all_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """Retrieves all documents from a Firestore collection, including their UUIDs."""
        try:
            docs = self.db.collection(collection_name).get()
            return [{**doc.to_dict(), 'uuid': doc.id} for doc in docs]
        except Exception as e:
            raise Exception(
                f"Error retrieving all documents from collection {collection_name}: {str(e)}"
            )

    def delete_document(self, document_id: str, collection_name: str) -> None:
        """Deletes a single document from Firestore."""
        try:
            doc_ref = self.db.collection(collection_name).document(document_id)
            doc_ref.delete()
        except Exception as e:
            raise Exception(f"Error deleting document {document_id}: {str(e)}")

    def bulk_delete_documents(self, document_ids: List[str], collection_name: str) -> None:
        """Deletes multiple documents from Firestore using batch operations."""
        try:
            batch = self.db.batch()
            for doc_id in document_ids:
                doc_ref = self.db.collection(collection_name).document(doc_id)
                batch.delete(doc_ref)
            batch.commit()
        except Exception as e:
            raise Exception(
                f"Error during bulk delete in collection {collection_name}: {str(e)}"
            )
