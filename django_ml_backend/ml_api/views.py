from django.shortcuts import render

# Create your views here.
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings

from .serializers import CTScanImageSerializer, MRIImageSerializer, XRayImageSerializer
from .models import CTScanImage, MRIImage, XRayImage
from .ml_models.ct_scan_model import CTScanModel
from .ml_models.mri_model import MRIModel
from .ml_models.xray_model import XRayModel
from rest_framework.permissions import AllowAny

# Initialize models (consider using lazy loading or singleton pattern for production)
ct_scan_model = CTScanModel()
mri_model = MRIModel()
xray_model = XRayModel()

class CTScanPredictionView(APIView):
    permission_classes = [AllowAny]  # Override default permissions to allow anyone
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        serializer = CTScanImageSerializer(data=request.data)
        if serializer.is_valid():
            # Save the uploaded image
            ct_scan_image = serializer.save()
            
            # Process the image with the model
            try:
                result = ct_scan_model.predict(ct_scan_image.image.path)
                
                # Update the database record with results
                ct_scan_image.processed = True
                ct_scan_image.result = result
                ct_scan_image.save()
                
                # Return the results with the image data
                return Response({
                    'image_id': ct_scan_image.id,
                    'uploaded_at': ct_scan_image.uploaded_at,
                    'result': result
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    'error': f"Error processing image: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class MRIPredictionView(APIView):
    permission_classes = [AllowAny]  # Override default permissions to allow anyone
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        serializer = MRIImageSerializer(data=request.data)
        if serializer.is_valid():
            # Save the uploaded image
            mri_image = serializer.save()
            
            # Process the image with the model
            try:
                result = mri_model.predict(mri_image.image.path)
                
                # Update the database record with results
                mri_image.processed = True
                mri_image.result = result
                mri_image.save()
                
                # Return the results with the image data
                return Response({
                    'image_id': mri_image.id,
                    'uploaded_at': mri_image.uploaded_at,
                    'result': result
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    'error': f"Error processing image: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class XRayPredictionView(APIView):
    permission_classes = [AllowAny]  # Override default permissions to allow anyone
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        serializer = XRayImageSerializer(data=request.data)
        if serializer.is_valid():
            # Save the uploaded image
            xray_image = serializer.save()
            
            # Process the image with the model
            try:
                result = xray_model.predict(xray_image.image.path)
                
                # Update the database record with results
                xray_image.processed = True
                xray_image.result = result
                xray_image.save()
                
                # Return the results with the image data
                return Response({
                    'image_id': xray_image.id,
                    'uploaded_at': xray_image.uploaded_at,
                    'result': result
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    'error': f"Error processing image: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# API status view to check if the service is running
class APIStatusView(APIView):
    permission_classes = [AllowAny] # Override default permissions
    def get(self, request, *args, **kwargs):
        return Response({
            'status': 'online',
            'version': '1.0.0',
            'models': {
                'ct_scan': 'loaded',
                'mri': 'loaded',
                'xray': 'loaded'
            }
        }, status=status.HTTP_200_OK)
