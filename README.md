# Image Denoising Machine Learning Model

## Description

The projects consists of a particular application of the Autoencoder, the Denoising Autoencoder. For the datasets on which the model was trained and tested on make sure to issue a request in the "issues" page of this repository and i will send you the complete archive with both datasets.

## Technologies Used

- PyTorch for all ML related code
- Matplotlib for some data visualisation

## How to Install the Project

1. Clone this repository
2. Navigate to the project directory
3. Install dependencies: `pip install -r requirements.txt`

## Testing

There is no need to have a GPU on the testing machine, the model is not configured to run on gpu as the laptop on which the tests were made did not have a dedicated GPU (model and data is on cpu)

I think the uploaded model has already been trained for a few epochs, but if you would like to start fresh then delete ```model_checkpoint.pt``` from the project and the model will reinitialize itself

After downloading the archive go to the folder's path and run ```python train.py``` in the terminal. You should now see the batches that the model utilizes to learn. More than that, at the end of each epoch the code outputs the Average Reconstruction Loss, making it visually easy to see how the model learns after each iteration.

## Known Issues

The model could use some fine-tuning regarding the optimiser and neural network itself, plus configuring the model and data to be on the gpu, not cpu (my peronal laptop does not have a dedicated GPU so i had to use the CPU)


