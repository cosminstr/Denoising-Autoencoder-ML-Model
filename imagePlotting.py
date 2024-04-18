from data_processing.data_preprocessing import *

#Plotting the images to test if they were loaded correctly

for images, labels in trainloader:

    denormalized_images = (images * 0.5) + 0.5
    numpy_images = denormalized_images.numpy()
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(4, 8, i)
        plt.imshow(numpy_images[i].transpose((1, 2, 0)))  
        plt.axis('off')
        class_name = trainset.classes[labels[i]]
        plt.title(f"{class_name}")
    plt.show()
    break 

for images, labels in trainnoisyloader:

    denormalized_images = (images * 0.5) + 0.5
    numpy_images = denormalized_images.numpy()
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(4, 8, i)
        plt.imshow(numpy_images[i].transpose((1, 2, 0)))  
        plt.axis('off')
        class_name = trainnoisyset.classes[labels[i]]
        plt.title(f"{class_name}")
    plt.show()
    break  



