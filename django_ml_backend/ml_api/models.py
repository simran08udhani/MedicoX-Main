from django.db import models

# Create your models here.
from django.db import models
import uuid
import os

def get_file_path(instance, filename):
    """Generate a unique file path for uploaded images."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    
    if isinstance(instance, CTScanImage):
        return os.path.join('ct_scans', filename)
    elif isinstance(instance, MRIImage):
        return os.path.join('mri', filename)
    elif isinstance(instance, XRayImage):
        return os.path.join('xray', filename)
    
    return filename

class CTScanImage(models.Model):
    image = models.ImageField(upload_to=get_file_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    result = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"CT Scan {self.id} - {self.uploaded_at}"

class MRIImage(models.Model):
    image = models.ImageField(upload_to=get_file_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    result = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"MRI {self.id} - {self.uploaded_at}"

class XRayImage(models.Model):
    image = models.ImageField(upload_to=get_file_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    result = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"X-Ray {self.id} - {self.uploaded_at}"