from piccolo.engine.sqlite import SQLiteEngine
from piccolo.conf.apps import AppRegistry

DB = SQLiteEngine(path="pixie.db")

APP_REGISTRY = AppRegistry(
    apps=[
        "piccolo.apps.user.piccolo_app",
        "piccolo_admin.piccolo_app",
    ]
)
