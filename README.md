# DPS-AI-Challenge
This is the challenge for DPS. I have built and deployed AI models in VERTEX AI.

### The challenge is to use the AUTO-MPG dataset and  Google’s managed ML platform Vertex AI to build an end-to-end machine learning workflow to predict the fuel efficiency of the given input array : test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0]

### The predicted fuel efficiency of the given test_mpg array [1, 2, 3, 2, -2, -1, -2, -1, 0] is 17.07827

# URL of endpoint : https://console.cloud.google.com/vertex-ai/locations/us-central1/endpoints/7822761348835573760?project=dps-auto-mpg

# Video Demo
In this video, I tried summarizing the whole process of this project.


https://user-images.githubusercontent.com/63927839/137320462-87188c27-00e9-4c76-ab9c-867797b7bc1d.mp4



##These are the steps followed in the process

# Step 1
To create an account in Google Cloud Console and set up the billing account.

# Step 2
Create a new project. Give the project a unique name. And once the project is launched, open the cloud console and authorize yourself using "gloud auth list" command.
![image](https://user-images.githubusercontent.com/63927839/137310775-c72a33f5-4538-4515-85d8-1e83674e1024.png)

Then run this command in the shell
"gcloud config list project"

and set your project using "gcloud config set project <unique-project-id>"
  
Cloud Shell has a few environment variables, including GOOGLE_CLOUD_PROJECT which contains the name of our current Cloud project. We'll use this in various places throughout this lab. You can see it by running "echo $GOOGLE_CLOUD_PROJECT"
  
  # Step 3
  Enable APIs
  In later steps, you'll see where these services are needed (and why), but for now, run this command to give your project access to the Compute Engine, Container Registry, and Vertex AI services
  ![image](https://user-images.githubusercontent.com/63927839/137311275-c65d0b69-6b3e-452d-a893-40084b957430.png)

  # Step 4
  Create a Cloud Storage Bucket
  To run a training job on Vertex AI, we'll need a storage bucket to store our saved model assets. Run the following commands in your Cloud Shell terminal to create a bucket:
 " BUCKET_NAME=gs://$GOOGLE_CLOUD_PROJECT-bucket"
  "gsutil mb -l us-central1 $BUCKET_NAME"
  
  The Bucket name must be unique.
  This is how you will see it in the container registry.
  ![image](https://user-images.githubusercontent.com/63927839/137311469-5d6ed01a-c421-4388-b004-f3ec93d3e003.png)
![image](https://user-images.githubusercontent.com/63927839/137311523-e491a9a4-53f5-4276-ab87-26ba91dc6b0a.png)

  # Step 5
  Set Alias for python using the following command
  ![image](https://user-images.githubusercontent.com/63927839/137311769-e5190b31-0f1e-4289-b9d7-05e7f99b2c88.png)

  # Step 6
  Containerize the code 
  To start, from the terminal in Cloud Shell, run the following commands to create the files we'll need for our Docker Container:
  
  mkdir mpg
  cd mpg
  touch Dockerfile
  mkdir trainer
  touch trainer/train.py
  
  You should now have an mpg/ directory that looks like the following:
  + Dockerfile
    + trainer/
      + train.py
  
  To view and edit these files, we'll use Cloud Shell's built-in code editor. You can switch back and forth between the editor and the terminal by clicking on the button on the   top right menu bar in Cloud Shell
  
  Create a Dockerfile
  
  To containerize our code we'll first create a Dockerfile. In our Dockerfile we'll include all the commands needed to run our image. It'll install all the libraries we're using   and set up the entry point for our training code.

  From the Cloud Shell file editor, open your mpg/ directory and then double-click to open the Dockerfile
  
  ![image](https://user-images.githubusercontent.com/63927839/137311999-7e3c4bf2-4db2-4fb4-85e1-c73c4453c159.png)
  
  Then copy the following into this file:


FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
WORKDIR /

#Copies the trainer code to the docker image.
COPY trainer /trainer

#Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]
  
  This Dockerfile uses the Deep Learning Container TensorFlow Enterprise 2.3 Docker image. The Deep Learning Containers on Google Cloud come with many common ML and data science frameworks pre-installed. The one we're using includes TF Enterprise 2.3, Pandas, Scikit-learn, and others. After downloading that image, this Dockerfile sets up the entrypoint for our training code, which we'll add in the next step.
  
  
  # Step 7 
  Add the custom model code in train.py. Check the code of train.py file for this.
  Once you've copied the code above into the mpg/trainer/train.py file, return to the Terminal in your Cloud Shell and run the following command to add your own bucket name to     the file:
  "sed -i "s|BUCKET_NAME|$BUCKET_NAME|g" trainer/train.py"
  
  # Step 8
   Build and test the container locally
  From your Terminal, run the following to define a variable with the URI of your container image in Google Container Registry:


  IMAGE_URI="gcr.io/$GOOGLE_CLOUD_PROJECT/mpg:v1"
  
  Then, build the container by running the following from the root of your mpg directory:


  docker build ./ -t $IMAGE_URI
  
  Once you've built the container, push it to Google Container Registry:


docker push $IMAGE_URI
  To verify your image was pushed to Container Registry, you should see something like this when you navigate to the Container Registry section of your console:
  ![image](https://user-images.githubusercontent.com/63927839/137312445-dd4e8dff-5738-414e-a490-84ee5dc60973.png)
  
  # Step 9
  Run a training job on Vertex AI. This step takes 15 minutes approximately.
  Vertex gives you two options for training models:

AutoML: Train high-quality models with minimal effort and ML expertise.
Custom training: Run your custom training applications in the cloud using one of Google Cloud's pre-built containers or use your own.
In this lab, we're using custom training via our own custom container on Google Container Registry. To start, navigate to the Training section in the Vertex section of your Cloud console:

![image](https://user-images.githubusercontent.com/63927839/137312579-f10104f5-1d6f-41fa-933a-cc710c8b83f0.png)
  
 ### Kick off the training job
Click Create to enter the parameters for your training job and deployed model:

Under Dataset, select No managed dataset
Then select Custom training (advanced) as your training method and click Continue.
Enter mpg (or whatever you'd like to call your model) for Model name
Click Continue
In the Container settings step, select Custom container:

Custom container option
  ![image](https://user-images.githubusercontent.com/63927839/137312733-5d39497a-85ab-4d6b-999d-6e4153b1a9f5.png)


In the first box (Container image), click Browse and find the container you just pushed to Container Registry. It should look something like this:
  ![image](https://user-images.githubusercontent.com/63927839/137312786-bb6ce8f5-a973-49b9-8eaf-fc8481ed08ca.png)


Find container

Leave the rest of the fields blank and click Continue.

We won't use hyperparameter tuning in this tutorial, so leave the Enable hyperparameter tuning box unchecked and click Continue.

In Compute and pricing, leave the selected region as-is and select n1-standard-4 as your machine type:
  ![image](https://user-images.githubusercontent.com/63927839/137312811-9e533a98-5861-4844-a28b-a340d5f7062d.png)


Under the Prediction container step, select No prediction container:
  ![image](https://user-images.githubusercontent.com/63927839/137312840-3e778bd8-8972-4498-9725-c09154511033.png)

  
  # Step 10
  Deploy a model endpoint.This takes approximately 10 mins.
  Here we'll be using the Vertex AI SDK to create a model, deploy it to an endpoint, and get a prediction.
  ###Install Vertex SDK
  From your Cloud Shell terminal, run the following to install the Vertex AI SDK:
    pip3 install google-cloud-aiplatform --upgrade --user
  ###Create model and deploy endpoint
  Next we'll create a Python file and use the SDK to create a model resource and deploy it to an endpoint. From the File editor in Cloud Shell, select File and then New File:
  ![image](https://user-images.githubusercontent.com/63927839/137313121-cefeecd7-59a8-46c7-bd02-ccbabc2757ea.png)
  Name the file deploy.py. Open this file in your editor and copy the following code:
  from google.cloud import aiplatform

'###Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="mpg-imported",
    artifact_uri="gs://io-vertex-codelab/mpg-model/",
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
)

### Deploy the above model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4"
)'
  
  Next, navigate back to the Terminal in Cloud Shell, cd back into your root dir, and run this Python script you've just created:
  cd ..
  python3 deploy.py | tee deploy-output.txt
  
  We're printing the logs from the above deploy command to a file so that we'll be able to use them in the next step when we make a prediction.
  ![image](https://user-images.githubusercontent.com/63927839/137313393-eb2671cb-b33c-4522-945a-4129d17b2866.png)
![image](https://user-images.githubusercontent.com/63927839/137313431-5a30fe37-6695-48df-96e7-4fa3eb60261b.png)

  In your Cloud Shell Terminal, you'll see something like the following log when your endpoint deploy has completed:


Endpoint model deployed. Resource name: projects/your-project-id/locations/us-central1/endpoints/your-endpoint-id
  
  
  # Step 11
  Get predictions on the deployed endpoint. 
  In your Cloud Shell editor, create a new file called predict.py:
  ![image](https://user-images.githubusercontent.com/63927839/137313537-ad3d4c2d-1387-423d-8444-3968a8260459.png)

  Enter the details in predict.py from the file predict.py.
  Next, go back to your Terminal and enter the following to replace ENDPOINT_STRING in the predict file with your own endpoint:
  ENDPOINT=$(cat deploy-output.txt | sed -nre 's:.*Resource name\: (.*):\1:p' | tail -1)
  sed -i "s|ENDPOINT_STRING|$ENDPOINT|g" predict.py
  
  Now it's time to run the predict.py file to get a prediction from our deployed model endpoint:
  python3 predict.py
  # Result is shown in the cloud shell
  # Predicted MPG:  17.07827
  
  ![image](https://user-images.githubusercontent.com/63927839/137313778-55130e08-2050-414d-ab00-c92a8934b97d.png)

  You should see the API's response logged, along with the predicted fuel efficiency for our test prediction.
  ![image](https://user-images.githubusercontent.com/63927839/137313895-cbd935ae-f20e-447a-a1d4-b7c891ac3960.png)
# 🎉 Congratulations! 🎉
 ## You have successfully implemented an end-to-end machine learning workflow using Google Vertex 
  
  After your use, please clean up the deployement by undeploying it.
  

  

