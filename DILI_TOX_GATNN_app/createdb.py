# -*- coding: utf-8 -*-

from gnn_app import app,db

app.app_context().push()
db.create_all()