# Hackathon README

This is how far our team managed to get through the challenge

## Installing Dependencies 

```pip install -r requirements.txt```

## How to test

There is no need to have a GPU on the testing machine, the model is not configured to run on gpu as the laptop on which the tests were made did not have a dedicated GPU (model and data is on cpu)

The model has already been trained for 20 epochs, if you would like to start fresh then delete ```model_checkpoint.pt``` from the project and the model will reinitialize itself

After downloading the archive go to the folder's path and run ```python train.py``` in the terminal. You should now see the batches that the model utilizes to learn. More than that, at the end of each epoch the code outputs the Average Reconstruction Loss, making it visually easy to see how the model learns after each iteration.


