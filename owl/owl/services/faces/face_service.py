import asyncio
import base64
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import io
import logging
import os
from typing import Dict, List, Tuple
import uuid

import aiohttp
from aiohttp import FormData
import boto3
from PIL import Image, ImageOps

from ...core.config import Configuration


logger = logging.getLogger(__name__)


####################################################################################################
# Person Identification
#
# Performs a search using FaceCheck.id.
####################################################################################################

TESTING_MODE = False

async def identify_person(config: Configuration, images: List[bytes]):
    site = 'https://facecheck.id'
    headers = {
        'Accept': 'application/json',
        'Authorization': config.face_check.api_key,
    }

    async with aiohttp.ClientSession() as session:
        form = FormData()
        for i, image_bytes in enumerate(images):
            form.add_field('images', image_bytes, filename=f"image-{i}.jpg", content_type='image/jpeg')
        
        try:
            # Post the image for initial processing/upload
            async with session.post(f"{site}/api/upload_pic", data=form, headers=headers) as response:
                response_data = await response.json()

                id_search = response_data['id_search']
                print(f"{response_data['message']} id_search={id_search}")

                json_data = {
                    'id_search': id_search,
                    'with_progress': True,
                    'status_only': False,
                    'demo': TESTING_MODE,
                }

                while True:
                    # Perform the search
                    async with session.post(f"{site}/api/search", json=json_data, headers=headers) as response:
                        response_data = await response.json()

                        if response_data.get('output'):
                            return response_data['output']['items']
                        
                        print(f"{response_data['message']} progress: {response_data['progress']}%")
                        await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Person identification: {e}")
            return None
        

####################################################################################################
# Face Detection and Recognition Service
#
# Detects and extracts faces, assigns unique IDs to them using AWS Rekognition.
####################################################################################################

@dataclass
class FaceMetrics:
    area: float
    eyes_open: bool
    eyes_open_confidence: float
    face_occluded: bool
    face_occluded_confidence: float
    yaw: float
    brightness: float
    sharpness: float

# Faces that are good enough to run through FaceCheck.id (upscaled already)
@dataclass
class FaceSamples:
    images: List[bytes] = field(default_factory=lambda: [])
    timestamps: List[datetime] = field(default_factory=lambda: [])

    def num_samples(self) -> int:
        return len(self.images)
    
    def add(self, image_bytes: bytes):
        self.timestamps.append(datetime.now())
        self.images.append(image_bytes)

    def time_since_first_image(self) -> float:
        if len(self.timestamps) <= 0:
            return 0
        return datetime.now() - self.timestamps[0]

class FaceService:
    def __init__(self, config: Configuration):
        self._config = config
        self._max_samples_per_face = 3
        self._face_samples_by_uuid = defaultdict(FaceSamples)
        self._client = boto3.client(
            'rekognition',
            region_name=config.aws.region_name,
            aws_access_key_id=config.aws.access_key,
            aws_secret_access_key=config.aws.secret_access_key
        )
        self._collection_id = config.aws.rekognition_collection_id
        self._create_collection_if_not_exists()

    #     os.makedirs("face_images", exist_ok=True)
    #     self._file_idx = 0
    #     self._html_fp = open("face_images/images.html", "w")
    #     self._html_fp.write("<html>\n<body>\n")

    # def __del__(self):
    #     self._html_fp.write("\n</body>\n</html>")
    #     self._html_fp.close()
    
    async def detect_faces(self, image_bytes: bytes) -> List[str]:
        # Pre-process image
        image = Image.open(io.BytesIO(image_bytes))
        rotated_image = image.rotate(90, expand=True)
        mirrored_image = ImageOps.mirror(rotated_image)  # This line mirrors the image horizontally
        
        # Detect largest face in image, if any
        cropped_image, metrics = self._detect_largest_face(image=mirrored_image)
        if cropped_image is None:
            return []

        # Scale the image x2 until it is larger than 2KB
        image_bytes = bytes()
        upscaled_image = cropped_image
        while len(image_bytes) < 2048:
            upscaled_image = upscaled_image.resize((upscaled_image.size[0] * 2, upscaled_image.size[1] * 2))
            image_bytes = self._get_image_bytes(image=upscaled_image)

        # Thresholds to reject poorly visible faces
        good = metrics.area >= 0.0188 and abs(metrics.yaw) <= 40 and metrics.brightness >= 45
        if not good:
            return []
        
        # Generate an HTML file
        # out_filename = f"face_images/{self._file_idx}.jpg"
        # with open(out_filename, "wb") as fp:
        #     fp.write(image_bytes)
        #     self._html_fp.write(f"<img src='{self._file_idx}.jpg'>\n")
        #     self._html_fp.write(f"<p>{metrics}</p>\n")
        # self._file_idx += 1

        # Detect face and index if needed
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
            if face_matches:
                # A known face was detected
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
                detected_face_ids.append(index_response["FaceRecords"][0]["Face"]["FaceId"])
        except self._client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "InvalidParameterException" and "no faces in the image" in e.response["Error"]["Message"]:
                pass
            else:
                logger.error(f"Error with AWS Rekognition: {e}")

        # At this point, detected_face_ids[0] holds our face, if it exists
        if len(detected_face_ids) > 0:
            face_id = detected_face_ids[0]

            # Store sample and if we reach the max, send for identification
            face_samples = self._face_samples_by_uuid[face_id]
            if face_samples.num_samples() < self._max_samples_per_face:
                face_samples.add(image_bytes=image_bytes)
                if face_samples.num_samples() == self._max_samples_per_face:
                    # Got sufficient samples, ready to ID
                    logger.info(f"SEARCHING PERSON {face_id}")
                    results = await identify_person(config=self._config, images=face_samples.images)
                    print(results)

        return detected_face_ids
    
    def _detect_largest_face(self, image: Image.Image) -> Tuple[Image.Image | None, FaceMetrics | None]:
        try:
            response = self._client.detect_faces(
                Image={ 'Bytes': self._get_image_bytes(image=image)  },
                Attributes=[ "ALL" ]
            )
            
            # Find largest face bounding box
            face_details = response["FaceDetails"]
            face_details_descending_size = sorted(face_details, reverse=True, key=lambda face_detail: face_detail["BoundingBox"]["Height"] * face_detail["BoundingBox"]["Width"])
            if len(face_details_descending_size) <= 0:
                return None, None
            face = face_details_descending_size[0]
            bbox = face["BoundingBox"]

            # Crop out the largest image, with 20% padding
            image_width, image_height = image.size
            horizontal_pad = 0.5 * bbox["Width"] * 0.25 * image_width
            vertical_pad = 0.5 * bbox["Height"] * 0.25 * image_height
            x1 = max(0, bbox["Left"] * image_width - horizontal_pad)
            x2 = min(image_width, x1 + bbox["Width"] * image_width + horizontal_pad)
            y1 = max(0, bbox["Top"] * image_height - vertical_pad)
            y2 = min(image_height, y1 + bbox["Height"] * image_height + vertical_pad)
            cropped = image.crop((x1, y1, x2, y2))
            cropped.save("cropped.jpg")

            # Get metrics
            face_metrics = FaceMetrics(
                area=bbox["Width"] * bbox["Height"],
                eyes_open=face["EyesOpen"]["Value"],
                eyes_open_confidence=face["EyesOpen"]["Confidence"],
                face_occluded=face["FaceOccluded"]["Value"],
                face_occluded_confidence=face["FaceOccluded"]["Confidence"],
                yaw=face["Pose"]["Yaw"],
                brightness=face["Quality"]["Brightness"],
                sharpness=face["Quality"]["Sharpness"]
            )

            return cropped, face_metrics

        except self.client.exceptions.ClientError as e:
            logger.error(f"Error with AWS Rekognition face detection: {e}")

        return None

    @staticmethod
    def _get_image_bytes(image: Image.Image) -> bytes:
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
