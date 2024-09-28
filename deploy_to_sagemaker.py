import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import time
import os

# Define your SageMaker execution role ARN here
sagemaker_execution_role = 'arn:aws:iam::746669191450:role/service-role/AmazonSageMaker-ExecutionRole-20240925T171407'  # Replace with your role ARN

# Define S3 paths
s3_model_uri = 's3://afiya-ml-s3/LoanStatus_artifacts.tar.gz'  # Path to the .tar.gz file

# Create a temporary directory to download the script
temp_dir = os.path.join(os.getenv('TEMP'), 'sagemaker_temp')
os.makedirs(temp_dir, exist_ok=True)

# Download inference.py from S3
print("Downloading inference.py from S3...")
local_script_path = os.path.join(temp_dir, 'inference.py')  # Update the path to the temporary directory
s3 = boto3.client('s3')
try:
    s3.download_file('afiya-ml-s3', 'inference.py', local_script_path)
    print("Downloaded inference.py successfully.")
except Exception as e:
    print(f"Failed to download inference.py: {e}")
    raise

# Use the specified execution role instead of getting it
role = sagemaker_execution_role
print(f"Using execution role: {role}")

# Create a SageMaker model using the Scikit-learn container
print("Creating SageMaker model...")
try:
    sklearn_model = SKLearnModel(
        model_data=s3_model_uri,
        role=role,
        entry_point=local_script_path,  # Local path to the downloaded inference.py
        framework_version='0.23-1'  # Framework version for Scikit-learn
    )
    print("SageMaker model created successfully.")
except Exception as e:
    print(f"Error creating SageMaker model: {e}")
    raise

# Deploy the model to an endpoint
print("Deploying the model to an endpoint...")
try:
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',  # You can adjust the instance size based on your needs
        wait=True  # Wait for the deployment to complete
    )
    print("Model deployed successfully. Endpoint name:", predictor.endpoint_name)
    
    # Optional: Monitor the endpoint status
    sm_client = boto3.client('sagemaker')
    endpoint_status = sm_client.describe_endpoint(EndpointName=predictor.endpoint_name)['EndpointStatus']
    
    while endpoint_status != 'InService':
        print(f"Waiting for endpoint to be in service... Current status: {endpoint_status}")
        time.sleep(30)  # Check status every 30 seconds
        endpoint_status = sm_client.describe_endpoint(EndpointName=predictor.endpoint_name)['EndpointStatus']
    
    print("Endpoint is now in service!")
except Exception as e:
    print(f"Error deploying the model: {e}")
    raise
