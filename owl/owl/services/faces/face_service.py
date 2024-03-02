import base64
import io
import logging
from typing import List
import uuid

import boto3
from PIL import Image, ImageOps

from ...core.config import AWSConfiguration

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(self, config: AWSConfiguration):
        self._client = boto3.client(
            'rekognition',
            region_name=config.region_name,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_access_key
        )
        self._collection_id = config.rekognition_collection_id
        self._create_collection_if_not_exists()
    
    def detect_faces(self, image_bytes: bytes) -> List[str]:
        # Pre-process image
        image = Image.open(io.BytesIO(image_bytes))
        rotated_image = image.rotate(90, expand=True)
        mirrored_image = ImageOps.mirror(rotated_image)  # This line mirrors the image horizontally
        
        # Detect largest face in image, if any
        image_bytes = self._detect_largest_face(image=mirrored_image)
        if image_bytes is None:
            return []
        
        # Index face and determine whether seen before
        detected_face_ids = []
        try:
            # Search for the face in the collection
            response = self._client.search_faces_by_image(
                CollectionId=self._collection_id,
                Image={'Bytes': image_bytes},
                FaceMatchThreshold=80,
                MaxFaces=1
            )
            face_matches = response.get('FaceMatches', [])
            known_face = False
            if face_matches:
                # A known face was detected
                known_face = True
                # Base64 encode the image bytes
                image_bytes_base64 = base64.b64encode(image_bytes).decode('utf-8')
                # Constructing the message
                message = {
                    "faceId": face_matches[0]['Face']['FaceId'],  # Assuming you want the first match's face ID
                    "imageBytes": image_bytes_base64
                }
                # Emit the message
                #await app_state.notification_service.emit_message(user.id, "known_face", message)
                logger.info(f"KNOWN FACE DETECTED {message['faceId']}")
                detected_face_ids.append(message["faceId"])
            else:
                # No known faces, let's index this new face
                index_response = self._client.index_faces(
                    CollectionId=self._collection_id,
                    Image={'Bytes': image_bytes},
                    ExternalImageId=str(uuid.uuid4()),  # Assuming you have imported uuid
                    MaxFaces=1,
                    QualityFilter="AUTO",
                    DetectionAttributes=['ALL']
                )
                known_face = False
        except self._client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "InvalidParameterException" and "no faces in the image" in e.response["Error"]["Message"]:
                pass
            else:
                logger.error(f"Error with AWS Rekognition: {e}")
        return detected_face_ids
    
    def _detect_largest_face(self, image: Image) -> bytes | None:
        try:
            response = self._client.detect_faces(
                Image={ 'Bytes': self._get_image_bytes(image=image)  },
                Attributes=[ "ALL" ]
            )
            
            # Find largest face bounding box
            face_details = response["FaceDetails"]
            face_details_descending_size = sorted(face_details, reverse=True, key=lambda face_detail: face_detail["BoundingBox"]["Height"] * face_detail["BoundingBox"]["Width"])
            if len(face_details_descending_size) <= 0:
                return None
            bbox = face_details_descending_size[0]["BoundingBox"]

            # Crop out the largest image
            image_width, image_height = image.size
            x1 = bbox["Left"] * image_width
            x2 = x1 + bbox["Width"] * image_width
            y1 = bbox["Top"] * image_height
            y2 = y1 + bbox["Height"] * image_height
            cropped = image.crop((x1, y1, x2, y2))
            #cropped.save("cropped.jpg")

            return self._get_image_bytes(image=cropped)

        except self.client.exceptions.ClientError as e:
            logger.error(f"Error with AWS Rekognition face detection: {e}")

        return None

    @staticmethod
    def _get_image_bytes(image: Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def _create_collection_if_not_exists(self):
        # Try to describe the collection to check if it exists
        try:
            response = self._client.describe_collection(CollectionId=self._collection_id)
            print(f"Collection '{self._collection_id}' already exists.")
        except self._client.exceptions.ResourceNotFoundException:
            # If the collection does not exist, create it
            response = self._client.create_collection(CollectionId=self._collection_id)
            print(f"Collection '{self._collection_id}' created.")
            print('Collection ARN:', response['CollectionArn'])
            print('Status code:', response['StatusCode'])
        except Exception as e:
            print(f"Error: {e}")
