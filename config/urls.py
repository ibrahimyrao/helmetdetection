from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.views.static import serve
import os

urlpatterns = [
    # Media ve Static dosyalarını manuel servis et (Gunicorn için)
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': str(settings.MEDIA_ROOT)}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': str(settings.STATIC_ROOT)}),
    
    path('admin/', admin.site.urls),
    path('', include('detector.urls')),
]
