# IndFood Image Classification

## Problem Statement
The rapid growth of digital technologies in the food industry has increased the demand for automated systems capable of accurately identifying food items from images. Manual food recognition and labeling are time-consuming, inconsistent and unsuitable for large-scale applications. Additionally, food images exhibit high variability in appearance due to differences in presentation, ingredients and, cultural or regional preparation styles. This project explores the potential of the Convolutional Neural Network(CNN) in identifying the food from the image samples. This aims to predict the name of the food item or dish or meal taken mainly in the Indian subcontinent region using the pre-trained model trained on images of food samples collected from various data sources. It supports data-driven decision-making, highlighting the need for dietary assessment, health management, and enhancing several food service applications.

## Dataset Description
![](/images/indfood_collage.jpg)
The dataset consists of around 135,335 images and 131 food items (classes). The images of food items belong to different food categories: sweet, curry, snacks, bread, cereal, beverage, pizza and kebab. The whole dataset is constructed from two open resources and split into train & val in an 85%-15% proportion. It can be accessed from this [link](https://github.com/soumen7saha/IndFood-Dataset).

## Dataset Analysis
- Categories
![](/images/food_categories.png)

- Class Distribution
![](/images/food_distribution.png)

- Hierarchial Sunburst
![](/images/food_hier_sunburst.png)

As the dataset is imbalanced and biased towards several food classes, a new balanced dataset with 25,000 training samples and 6,000 validation samples, is achieved through data augmentation techniques to feed into the CNN model for training.

## Model Training & Metrics
| Model | Hyperparameters | Top-1 Val Accuracy | Checkpoint | Test Accuracy |
| --- | --- | --- | --- | --- |
| MobileNetV2 | epochs=50, learning_rate=0.001, size_inner=1000, droprate=0.4 | 62.3% | [food_mobilenet_v12_19_0.623.pth](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/models/food_mobilenet_v12_19_0.623.pth) | 37.93% |
| EfficientNet-V2-S | epochs=40, learning_rate=0.001, size_inner=500, droprate=0.3 | 70.7% | [food_effnet_v23_40_0.707.pth](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/models/food_effnet_v23_40_0.707.pth) | 55.17% |
| ConvNeXT-S | epochs=50, learning_rate=0.001, size_inner=1000, droprate=0.3 | 83.8% | [food_cnext_v33_38_0.838.pth](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/models/food_cnext_v33_38_0.838.pth) | 68.97% |
| ResNet-152 | epochs=15, learning_rate=0.001, unfrozen_layers=2 | 88.7% | [food_resnet_v42_12_0.887.pth](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/models/food_resnet_v42_12_0.887.pth) | 79.31% | 

Different pre-trained CNN models are trained using PyTorch and evaluated on the basis of accuracy and their performance on the validation set. The top two models with the maximum test/val accuracy are selected to deploy in the application and exported to the [models](https://github.com/soumen7saha/IndFood-Image-Classification/tree/main/models) sub-folder.

## Project Folder Structure
```
└── 📁IndFood-Image-Classification
    └── 📁.venv
        └── 📁bin
            ├── activate
    └── 📁data
        └── 📁test
        └── 📁train
        └── 📁val
    └── 📁images
    └── 📁k8s
        ├── deployment.yaml
        ├── hpa.yaml
        ├── service.yaml
    └── 📁models
        ├── food_classifier_convnexts_v2.onnx
        ├── food_classifier_convnexts_v2.onnx.data
        ├── food_resnet_v42_12_0.887.pth
        ├── resnet152.pth
    └── 📁notebooks
        ├── Model Testing.ipynb
        ├── Model Training ConvNeXT-S.ipynb
        ├── Model Training ResNet152.ipynb
    └── 📁src
        └── 📁images
            ├── masala_dosa.jpg
        └── 📁scripts
            ├── model.py
            ├── predict.py
            ├── serve.py
            ├── train.py
    └── 📁static
        └── 📁uploads
    └── 📁templates
        ├── index.html
    └── 📁tests
        ├── load_test.py
        ├── test.py
    ├── .python-version
    ├── app.py
    ├── main.py
    ├── Dockerfile
    ├── pyproject.toml
    ├── requirements.txt
    └── uv.lock
```
- .venv : manages the project's isolated virtual environment
- data : stores the dataset files and links
- models/model : stores the exported model files
- notebooks : contains the notebook files used to trian and test the models
- src/scripts : contains the python scripts file segragated from the notebook 
- app.py : entry point to run the project
- main.py : entry point to run the integrated streamlit application
- uv.lock : used to install all the specified packages into the project's virtual environment
- Dockerfile : used to build the docker container
- k8s : contains kubernetes related yaml files
- tests : stores python script file to test the API endpoints and load balancing

## How to run locally?
- Clone the Project from git
    
        git clone https://github.com/soumen7saha/IndFood-Image-Classification.git

- Make sure you have the python version specified in [pyproject.toml](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/pyproject.toml) or [.python-version](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/.python-version) file and pip (pip installs packages) in the local system

- Install uv using pip

        pip install uv

- Go to the project directory, open terminal & run

        uv sync

- Activate the uv environment
    
        source .venv/bin/activate

- Make sure the port _9696_ is not allocated to any process
    
- Run
    
        uv run python app.py

## How to run via Docker?
- Go to the project directory in terminal and run the following commands:

        cat Dockerfile
        docker build -t indfood-imgclassification:v3 .
        docker run -it --rm -p 9696:9696 indfood-imgclassification:v3

## API Usage Examples
- Move to the project directory and open terminal
- Run the following commands in terminal (in separate tabs/windows)
	
		# command
        python ./tests/test.py
		
        # output
        _______________________________________
        Status Code: 200
        Response Text: '{"t1_class":"masala_dosa","t5_preds":{"masala_dosa":-13.76608657836914,"idli":-14.426803588867188,"pav_bhaji":-17.09644317626953,"chicken_tikka_masala":-17.372220993041992,"anda_curry":-19.438465118408203}}'
        _______________________________________

        Top predicted class: masala_dosa

        Top 5 predictions:
        masala_dosa
        idli
        pav_bhaji
        chicken_tikka_masala
        anda_curry

        # api call
        curl -X 'POST' \
        'http://localhost:9696/predict_food' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "img_url": "data/test/biryani.jpg",
        "model": "resnet"
        }'

        # api response
        {
        "t1_class": "biryani",
        "t5_preds": {
            "biryani": 1.8974436521530151,
            "rajma_chawal": -14.073193550109863,
            "poha": -15.057899475097656,
            "palak_paneer": -15.629433631896973,
            "chicken_wings": -16.865856170654297
        }
        }

        # api call
        curl -X 'POST' \
        'http://localhost:9696/predict_food' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "img_url": "data/test/biryani.jpg",
        "model": "convns"
        }'

        # api response
        {
        "t1_class": "biryani",
        "t5_preds": {
            "biryani": -2.1676177978515625,
            "masala_papad": -8.092710494995117,
            "rajma_chawal": -8.586798667907715,
            "papdi_chaat": -13.189144134521484,
            "anda_curry": -13.448341369628906
        }
        }

## User Interface
- Homepage
![](/images/0.png)

- Image File Upload
![](/images/1.png)

- Food Class Output
![](/images/2.png)

## K8S Deployment
- Goto the k8s directory

        cd k8s

- Create a cluster with name _indfoodic_

        kind create cluster --name indfoodic
        kubectl cluster-info

- Check if the node is ready, you should see one node in "Ready" status.

        kubectl get nodes

- Load image to kind

        kind load docker-image indfood-imgclassification:v3 --name indfoodic

- Apply deployment.yaml & check the deployments and pods

        kubectl apply -f deployment.yaml
        kubectl get deployments
        kubectl get pods

- Check the deployment information

        kubectl describe deployment indfood-imgclassification

- Create the service & check the service information

        kubectl apply -f service.yaml
        kubectl get services
        kubectl describe service indfood-imgclassification

- View logs & make test api call

        kubectl logs -l app=indfood-imgclassification --tail=20
        curl http://localhost:30080/health

- Testing the Deployed Service

        kubectl port-forward service/indfood-imgclassification 30080:9696

- Testing Autoscaling

        kubectl apply -f hpa.yaml
        kubectl get hpa
        kubectl describe hpa indfood-imgclassification-hpa

        cd tests
        uv run python load_test.py

        # command output
        Starting load test...
        Watch HPA with: kubectl get hpa -w
        Watch pods with: kubectl get pods -w
        Sending 1000 requests with 50 concurrent workers

        Load test complete!
        Duration: 1.83 seconds
        Requests per second: 547.43
        Successful requests: 1000
        Failed requests: 0

## Streamlit Cloud Deployment
To deploy in the streamlit cloud, add the git repository link and the entry file ([main.py](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/main.py)). You will be provided with the public URL - https://indfood-image-classification.streamlit.app/ 

A demo video of the streamlit application is provided [here](https://github.com/soumen7saha/IndFood-Image-Classification/blob/main/images/demo_ific.webm):

https://github.com/user-attachments/assets/d69c1370-0a22-46b6-bb17-5f1f137ba62a

## Architecture Diagram
![](/images/arch_diag.png)

## Known Limitations & Next Steps
The constructed dataset suffers from class imbalance with several food categories underrepresented that leads to potential bias. The resnet-152 model, even though it gave an accuracy of 88%, is trained for lesser epochs (15) and on a smaller balanced dataset with 25,000 samples due to resource constraints. These challenges could be addressed in subsequent research, potentially through the application of alternative models like vision transformers or multi-modal LLMs. The current application could be integrated with modern frontend frameworks like ReactJS and deployed in comprehensive, on-demand cloud computing platform like AWS.
