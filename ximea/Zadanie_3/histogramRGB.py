import cv2
import matplotlib.pyplot as plt

img_path = "./ximea/Zadanie_3/images/cheetah.jpg"

img = cv2.imread(img_path)

if img is None:
    print("Error: Unable to load the image.")
else:
    # Plotting function for histogram
    def plot_histogram(histogram, color):
        intensity_values = range(256)
        plt.bar(intensity_values, histogram, color=color)
        plt.title('Channel Histogram')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.show()

    # Calculate histogram for a single channel
    def calculate_histogram(image_data, channel):
        histogram = [0] * 256  # Initialize histogram with zeros
        # Iterate over the image pixels
        for row in image_data:
            for pixel in row:
                intensity = pixel[channel]  # Get the intensity value for the specified channel
                histogram[intensity] += 1
        return histogram

    # Calculate histograms for each channel
    histogram_red = calculate_histogram(img, 2)  # Red channel (BGR order)
    histogram_green = calculate_histogram(img, 1)  # Green channel (BGR order)
    histogram_blue = calculate_histogram(img, 0)  # Blue channel (BGR order)

    # Plot histograms
    plot_histogram(histogram_red, 'red')
    plot_histogram(histogram_green, 'green')
    plot_histogram(histogram_blue, 'blue')

    # Convert BGR to RGB format for plotting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Display the image
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
