Panoramic Image Optical Flow Project.

# File Structure

The project file structure are like following image:

```
.
├── code
|   ├── README.md
│   └── python                         # the python used to load & visual & Evaluation
│
├── data                               # experiment data
|   ├── README.md
│  
|
├── doc
|   ├── README.md
|   ├── paper_draft                            # the images used in the documents
│   └── optical_flow_ground_truth.md   
|
└── README.md
```

Please read the README.md file in each folder to get more information.


# Dataset Generate

The Replica-Dataset code generate the raw data, the `code/python/replica360/` code to post-process the data to generate the final dataset data.

The core method implement in `code/python/utility`.