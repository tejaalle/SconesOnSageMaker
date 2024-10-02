import json
# import sagemaker
import base64
import boto3

# from sagemaker.serializers import IdentitySerializer
# from sagemaker.predictor import Predictor

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-09-27-22-06-59-974"


def lambda_handler(event, context):
    current_event = event["body"]

    # Decode the image data
    image = base64.b64decode(current_event["image_data"])

    # Instantiate a Predictor
    sagemaker_runtime = boto3.client('runtime.sagemaker')
    predictor = sagemaker_runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                                  ContentType='application/x-image',
                                                  Body=image)

    # For this model the IdentitySerializer needs to be "image/png"
    # predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor['Body'].read()

    # We return the data back to the Step Function
    current_event["inferences"] = json.loads(inferences.decode('utf-8'))
    return {
        'statusCode': 200,
        'body': current_event
    }
