import os
import pathlib
import path
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin
from firebase_admin.db import Reference
from typing import Dict


class FireBase:
    firebase_service_path: str = ""
    authentication_path: str = ""
    db_name: str = ""
    database_reference:Reference = None
    company_table_reference:Reference = None

    def __init__(self, firebase_service_path: str, authentication_path: str, db_name: str, company_tag: str):
        self.firebase_service_path = firebase_service_path
        self.authentication_path = authentication_path
        self.db_name = db_name
        self.company_tag = company_tag

    def connection(self):
        database_reference = db.reference(self.db_name)
        self.company_table_reference = database_reference.child(self.company_tag)

    def insert_data(self, summarized_datasource:Dict[str,Dict[str]]):
        self.company_table_reference.set(summarized_datasource)

