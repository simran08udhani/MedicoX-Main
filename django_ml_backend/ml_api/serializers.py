from rest_framework import serializers
from .models import CTScanImage, MRIImage, XRayImage

class CTScanImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = CTScanImage
        fields = ['id', 'image', 'uploaded_at', 'processed', 'result']
        read_only_fields = ['uploaded_at', 'processed', 'result']

class MRIImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = MRIImage
        fields = ['id', 'image', 'uploaded_at', 'processed', 'result']
        read_only_fields = ['uploaded_at', 'processed', 'result']

class XRayImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = XRayImage
        fields = ['id', 'image', 'uploaded_at', 'processed', 'result']
        read_only_fields = ['uploaded_at', 'processed', 'result']