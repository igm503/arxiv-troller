"""
WSGI config for arxiv_troller project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from django.db.backends.signals import connection_created

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "arxiv_troller.settings")

application = get_wsgi_application()


def configure_web_connection(connection, **kwargs):
    """Apply settings.WEB_DB_SESSION to each database connection a web worker opens."""
    from django.conf import settings

    with connection.cursor() as cursor:
        for name, value in settings.WEB_DB_SESSION.items():
            cursor.execute("SELECT set_config(%s, %s, false)", [name, value])


connection_created.connect(configure_web_connection)
