import numpy as np
import matplotlib.pyplot as plt
from q1_script import read_labels, read_pixels
from PCAhelpers import calculate_covariance_matrix, calculate_eig_sorted

#part 1 reading images and labels
images = read_pixels("data/train-images-idx3-ubyte.gz")
labels = read_labels("data/train-labels-idx1-ubyte.gz")

#In reality we already know these so it might be better to just hardcode them to save up in time
number_of_images = labels.shape[0]
number_of_pixels = images.shape[0]
number_of_pixels_per_image = number_of_pixels // number_of_images

#part 2: normalization
pixels = images.reshape(number_of_images, number_of_pixels_per_image)
mean_centered_pixels = pixels - pixels.mean(axis = 0, keepdims=True) 

#part 3: calculating the covariance matrix
covariance_matrix = calculate_covariance_matrix(mean_centered_pixels) 

#part 4: finding eigenvalues and eigenvectors, sorting them in decreasing order(such that the highest values come first)
eigenvalues, eigenvectors = calculate_eig_sorted(covariance_matrix)

#part 5 finding PVE(proportion of variance explained): these add up to 1
PVE = eigenvalues / eigenvalues.sum()

#For the first ten, value is:
for i in range(10):
    print(PVE[i])

print("\n")

#for at least 0.7 of the data, we need:
i = 1
#? are the sums not in increasing order
while PVE[:i].sum() < 0.7:
    print(PVE[:i].sum())
    i += 1

print(PVE[:i].sum())
print(f"We require {i} Principal Components")

#First 10 principal component vectors:
top_10_eigenvectors = eigenvectors[:, :10]
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i in range(10):
    eigenvector_image = top_10_eigenvectors[:,i].reshape(28,28)
    axes[i].imshow(eigenvector_image, cmap="Greys_r")
    axes[i].axis("off")
    axes[i].set_title(f"PC {i+1}")
plt.show()

#top 2 
top_2_eigenvectors = eigenvectors[:, :2]
projected_data = np.dot(mean_centered_pixels[:100], top_2_eigenvectors)

colors = plt.cm.rainbow(np.linspace(0, 1, 10))  
for i in range(10):
    indices = labels[:100] == i
    plt.scatter(projected_data[indices, 0], projected_data[indices, 1], color=colors[i], label=f'Digit {i}')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
#plt.title('Plot')
plt.legend()
plt.show()


#PCA on top k
def compress_image(sorted_eigenvectors, unprocessed_image, k):
    mean_vector = np.mean(unprocessed_image)
    centered_image = unprocessed_image - mean_vector

    top_k_vectors = sorted_eigenvectors[:, :k]
    projected_data = np.dot(centered_image, top_k_vectors)
    print(projected_data.shape)
    reconstructed_data = np.dot(projected_data, top_k_vectors.T) + mean_vector
    print(reconstructed_data.shape)
    return reconstructed_data.reshape((28, 28))

img_to_compress = pixels[:1] #first image

plt.figure(figsize=(8, 4))

plt.subplot(3, 3, 1)
plt.imshow(img_to_compress.reshape(28, 28), cmap="Greys_r")
plt.title("Original")

plt.subplot(3, 3, 2)
plt.imshow(compress_image(eigenvectors, img_to_compress, 784), cmap="Greys_r")
plt.title("Reconstructed exactly(784)") #exact same image, seems to be working correctly

plt.subplot(3, 3, 3)
plt.imshow(compress_image(eigenvectors, img_to_compress, 1), cmap="Greys_r")
plt.title("Reconstructed From 1 Principal Component") 

plt.subplot(3, 3, 4)
plt.imshow(compress_image(eigenvectors, img_to_compress, 50), cmap="Greys_r")
plt.title("Reconstructed From 50 Principal Components") 

plt.subplot(3, 3, 5)
plt.imshow(compress_image(eigenvectors, img_to_compress, 100),cmap="Greys_r")
plt.title("Reconstructed From 100 Principal Components") 

plt.subplot(3, 3, 6)
plt.imshow(compress_image(eigenvectors, img_to_compress, 250), cmap="Greys_r")
plt.title("Reconstructed From 250 Principal Components") 

plt.subplot(3, 3, 7)
plt.imshow(compress_image(eigenvectors, img_to_compress, 500),cmap="Greys_r")
plt.title("Reconstructed From 500 Principal Components") 

plt.subplots_adjust(wspace=3) 
plt.show()