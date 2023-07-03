import os
import pathlib
import path
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin
from firebase_admin import firestore
from typing import Dict, Any, List

CURRENT_PATH = pathlib.Path(os.getcwd())
ROOT_PATH = CURRENT_PATH.parent.parent.absolute()
Authentication_PATH = os.path.join(ROOT_PATH, os.sep.join(["credentials", "serviceAccountKey.json"]))
DATABASE_URL = "https://atssa-a5257-default-rtdb.firebaseio.com/"


class FireBase:
    firebase_service_path: str = ""
    authentication_path: str = ""
    db: Any = ""
    collection_name: str = None

    def __init__(self):
        self.authentication_path = Authentication_PATH
        self.create_firebase_admin_if_not_exist()

    def insert_summarized_data(self, company_tag, summarized_data:Dict[str,str]):
        self.create_firebase_admin_if_not_exist()
        ref = db.reference(company_tag).child('summarized_data ')
        for ds_id, data in summarized_data.items():
            ref.child(ds_id).set(data)

    def create_firebase_admin_if_not_exist(self):
        cred = credentials.Certificate(self.authentication_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})

    def insert_metadata_data(self, metadata, company_tag: str):
        self.create_firebase_admin_if_not_exist()
        ref = db.reference(company_tag)
        metadata_ref = ref.child('metadata')
        for ds_id, md in metadata.items():
            metadata_ref.child(ds_id).set(md)
