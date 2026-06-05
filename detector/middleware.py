import json
import os
from django.shortcuts import render

class MaintenanceMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith('/admin') or request.path.startswith('/static'):
            return self.get_response(request)

        shared_file_path = '/app/shared_status/projects_status.json'
        if os.path.exists(shared_file_path):
            try:
                with open(shared_file_path, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                
                my_domain = os.environ.get('MAINTENANCE_DOMAIN', 'helmet.yourdomain.com')
                if status_data.get(my_domain) == 'MAINTENANCE':
                    return render(request, 'maintenance.html', status=503)
            except:
                pass
        
        return self.get_response(request)
