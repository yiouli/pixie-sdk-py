from piccolo.engine.sqlite import SQLiteEngine
from piccolo.conf.apps import AppRegistry

DB = SQLiteEngine(path="pixie.db")

apps = []

try:
    import piccolo_admin

    apps.append("piccolo.apps.user.piccolo_app")
    apps.append("piccolo_admin.piccolo_app")
except ImportError:
    pass


APP_REGISTRY = AppRegistry(apps=apps)
