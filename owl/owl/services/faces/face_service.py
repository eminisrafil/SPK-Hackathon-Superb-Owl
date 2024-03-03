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
from ..llm.llm_service import LLMService

import aiohttp
from aiohttp import FormData
import boto3
import cv2
from PIL import Image, ImageOps
from bs4 import BeautifulSoup
import json
from pydantic import BaseModel, RootModel
import numpy as np

from ...core.config import Configuration


logger = logging.getLogger(__name__)



####################################################################################################
# Person Identification
#
# Performs a search using FaceCheck.id.
####################################################################################################

TESTING_MODE = False

class PersonResult(BaseModel):
    guid: str
    score: int
    group: int
    base64: str
    url: str
    index: int

class PersonResults(RootModel):
    root: List[PersonResult]

async def identify_person(config: Configuration, images: List[bytes]) -> List[PersonResult]:
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
                if not id_search:
                    return []
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
                            try:
                                return PersonResults.model_validate(obj=response_data["output"]["items"]).root
                            except:
                                return []
                        
                        print(f"{response_data['message']} progress: {response_data['progress']}%")
                        await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Person identification: {e}")
            return []
        

####################################################################################################
# Face Detection and Recognition Service (Implementation)
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
    cv_sharpness: float

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

class _FaceService:
    def __init__(self, config: Configuration):
        self._config = config
        self._llm_service = LLMService(config.llm)
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

        os.makedirs("face_images", exist_ok=True)
        self._file_idx = 0
        self._html_fp = open("face_images/images.html", "w")
        self._html_fp.write("<html>\n<body>\n")

    def __del__(self):
        self._html_fp.write("\n</body>\n</html>")
        self._html_fp.close()

    async def scrape(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url_data(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return results
    
    async def fetch_url_data(self, session, url):
        try:
            # Adjust the API URL with your parameters
            api_url = "https://app.scrapingbee.com/api/v1/"
            params = {
                "render_js": "false",
                "extract_rules": "{\"text\": \"body\"}",
                "url": url,
                "api_key": self._config.scraping_bee.api_key
            }

            async with session.get(api_url, params=params) as response:
                response.raise_for_status()
                data = await response.json() 
                
                text = data.get('text', '')  
                
                return {
                    'url': url,
                    'text': text
                }
        except Exception as e:
            print(f'Error fetching {url}: {e}')
            return {'url': url, 'error': str(e)}
        
    async def detect_faces(self, image_bytes: bytes) -> List[PersonResult]:
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

        # Generate an HTML file
        out_filename = f"face_images/{self._file_idx}.jpg"
        with open(out_filename, "wb") as fp:
            fp.write(image_bytes)
            self._html_fp.write(f"<img src='{self._file_idx}.jpg'>\n")
            self._html_fp.write(f"<p>{metrics}</p>\n")
        self._file_idx += 1

        # Thresholds to reject poorly visible faces
        good = metrics.area >= 0.0077 and abs(metrics.yaw) <= 40 and metrics.brightness >= 35 and metrics.eyes_open and metrics.cv_sharpness >= 35
        if not good:
            logger.error(f"Failed: {metrics}")
            return []

        # Detect face and index if needed
        face_id = None
        try:
            face_id = self._identify_face(image_bytes = image_bytes)

            # If found, send update
            if face_id is not None:
                # image_bytes_base64 = base64.b64encode(image_bytes).decode('utf-8')
                # message = {
                #     "faceId": face_id,  # Assuming you want the first match's face ID
                #     "imageBytes": image_bytes_base64
                # }
                #await app_state.notification_service.emit_message(user.id, "known_face", message)
                logger.info(f"KNOWN FACE DETECTED {face_id}")
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
                face_id = index_response["FaceRecords"][0]["Face"]["FaceId"]
        except self._client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "InvalidParameterException" and "no faces in the image" in e.response["Error"]["Message"]:
                pass
            else:
                logger.error(f"Error with AWS Rekognition: {e}")

        # Handle face
        if face_id is not None:
            # Store sample and if we reach the max, send for identification
            face_samples = self._face_samples_by_uuid[face_id]
            if face_samples.num_samples() < self._max_samples_per_face:
                face_samples.add(image_bytes=image_bytes)
                if face_samples.num_samples() == self._max_samples_per_face:
                    # Got sufficient samples, ready to ID
                    logger.info(f"SEARCHING PERSON {face_id}")
                    person_results = await self._identify_person(target_face_id=face_id, images=face_samples.images)
                    if person_results:
                        urls_to_fetch = [item.url for item in person_results]
                        urls_data = await self.scrape(urls_to_fetch)
                        formatted_urls_data = json.dumps(urls_data, ensure_ascii=False, indent=2)
                        tools = [{
                            "type": "function",
                            "function": {
                                "name": "person_prediction",
                                "description": "Given a list of web data, predict the most likely person based on the content of the pages.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Full Name of the person of most likely target person.",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Description of the person of most likely target person. keep it short in highlighting the most important features of the person.",
                                        },

                                    },
                                    "required": ["name", "description"],
                                },
                            },
                        }]
                        llm_result = await self._llm_service.async_llm_completion(
                            tools=tools,
                            #tool_choice={"name": "person_prediction"},
                            messages=[
                                {"content": f"You are the world's most advanced person identifier. we have a set of webpages. some may contain our target or some may be irrelevant. you take results from a web search and try and identify the most likely individual given that some result may be irrelevant based on that we are in a tich event in san Francisco. this is a person your client met while at a hackathon in san Francisco. you must help him find the person because it's important for human flourishing.", "role": "system"},
                                {"content": f"Web data: {formatted_urls_data}", "role": "user"}
                            ])
                        print(llm_result)
                    print(person_results)

        return []
    
    async def _identify_person(self, target_face_id: str, images: List[bytes]) -> List[PersonResult]:
        """
        Attempts to identify a person given a list of images of the same person. Returns zero or
        more person results from FaceCheck.
        """

        for i, image_bytes in enumerate(images):
            with open(f"submitted_{target_face_id}_{i}.webp", mode="wb") as fp:
                fp.write(image_bytes)

        results = await identify_person(config=self._config, images=images)

        # Check each person against our Amazon database to see whether it is the intended target
        validated_results = []
        for person_result in results:
            # Get base64 data and convert to bytes. Base64 data is encoded as e.g.:
            # data:image/webp;base64, <data>
            parts = person_result.base64.split(" ")
            if len(parts) != 2:
                continue
            image_base64 = parts[1]

            # Convert from webp -> JPEG for Amazon
            webp_image_bytes = base64.b64decode(image_base64)
            webp_image_buffer = io.BytesIO(webp_image_bytes)
            image_bytes = self._get_image_bytes(image=Image.open(webp_image_buffer))

            # Try to look up face in AWS (to validate that the person result from FaceCheck is the
            # same person we sent)
            face_id = self._identify_face(image_bytes=image_bytes, face_match_threshold=80)
            if face_id == target_face_id:
                validated_results.append(person_result)
            
        return validated_results

    def _identify_face(self, image_bytes: bytes, face_match_threshold: int = 80) -> str | None:
        """
        Identify a face. If this is a known face, its ID (a UUID) is returned. This identifies faces
        belonging to the same person but does not establish that person's real-world identity.
        """
        # Search for the face in the collection
        response = self._client.search_faces_by_image(
            CollectionId=self._collection_id,
            Image={'Bytes': image_bytes},
            FaceMatchThreshold=face_match_threshold,
            MaxFaces=1
        )
        face_matches = response.get('FaceMatches', [])
        if face_matches:
            # A known face was detected
            return face_matches[0]['Face']['FaceId']
        return None

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
            #cropped.save("cropped.jpg")

            # Compute our own sharpness metric
            gray = cv2.cvtColor(np.array(cropped), cv2.COLOR_BGR2GRAY)
            cv_sharpness = cv2.Laplacian(src=gray, ddepth=cv2.CV_64F).var()

            # Get metrics
            face_metrics = FaceMetrics(
                area=bbox["Width"] * bbox["Height"],
                eyes_open=face["EyesOpen"]["Value"],
                eyes_open_confidence=face["EyesOpen"]["Confidence"],
                face_occluded=face["FaceOccluded"]["Value"],
                face_occluded_confidence=face["FaceOccluded"]["Confidence"],
                yaw=face["Pose"]["Yaw"],
                brightness=face["Quality"]["Brightness"],
                sharpness=face["Quality"]["Sharpness"],
                cv_sharpness=cv_sharpness
            )

            return cropped, face_metrics

        except self._client.exceptions.ClientError as e:
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


####################################################################################################
# Inter-process Communication
####################################################################################################

@dataclass
class DetectFacesCommand:
    image_bytes: bytes

@dataclass
class DetectFacesResult:
    persons: List[PersonResult]

@dataclass
class TerminateProcessCommand:
    pass

####################################################################################################
# Face Service (Public)
####################################################################################################

from multiprocessing import Queue, Process
from ...core.utils import AsyncMultiprocessingQueue

class FaceService:
    def __init__(self, config: Configuration, notification_service):
        self._notification_service = notification_service
        self._request_queue = AsyncMultiprocessingQueue(queue=Queue())
        self._response_queue = AsyncMultiprocessingQueue(queue=Queue())
        # Start process
        process_args = (
            self._request_queue.underlying_queue(),
            self._response_queue.underlying_queue(),
            config
        )
        self._process = Process(target=FaceService._run, args=process_args)
        self._process.start()

    def __del__(self):
        self._request_queue.underlying_queue().put(TerminateProcessCommand())
        self._process.join()

    async def detect_faces(self, image_bytes: bytes) -> List[PersonResult]:
        await self._request_queue.put(DetectFacesCommand(image_bytes=image_bytes))
        response = await self._response_queue.get()
        if isinstance(response, DetectFacesResult):
            #TODO: send notification here
            return response.persons
        return []
    
    def _run(request_queue: Queue, response_queue: Queue, config: Configuration):
        face_service = _FaceService(config=config)

        while True:
            request = request_queue.get()
            
            # Command: terminate process
            if isinstance(request, TerminateProcessCommand):
                break

            # Command: detect face
            if isinstance(request, DetectFacesCommand):
                request: DetectFacesCommand = request
                persons = asyncio.run(face_service.detect_faces(image_bytes=request.image_bytes))
                response_queue.put(DetectFacesResult(persons=persons))
        
        print("TERMINATED FACE SERVICE PROCESS")