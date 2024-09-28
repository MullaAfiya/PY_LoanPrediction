import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.model import SKLearnModel
import time

# Define S3 paths
s3_model_uri = 's3://afiya-ml-s3/LoanStatus_artifacts.tar.gz'  # Path to the .tar.gz file

# Download inference.py from S3
print("Downloading inference.py from S3...")
local_script_path = '/tmp/inference.py'
s3 = boto3.client('s3')
try:
    s3.download_file('afiya-ml-s3', 'inference.py', local_script_path)
    print("Downloaded inference.py successfully.")
except Exception as e:
    print(f"Failed to download inference.py: {e}")
    raise

# Get the execution role
print("Getting execution role...")
try:
    role = get_execution_role()
    print(f"Execution role: {role}")
except Exception as e:
    print(f"Failed to get execution role: {e}")
    raise

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

