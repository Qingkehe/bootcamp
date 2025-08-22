#!/usr/bin/env python3
"""
Image Processing Program for Beginners
=====================================

This program teaches you how to:
1. Load and work with images in Python
2. Apply different types of image effects (blur, mosaic)
3. Understand what happens to images when we process them
4. Use statistics to measure image changes
5. Save your processed images

What is an Image?
-----------------
An image is just a grid of numbers! Each number represents how bright a pixel is:
- 0 = completely black
- 255 = completely white  
- Numbers in between = different shades of gray

What is a Histogram?
--------------------
A histogram shows how many pixels have each brightness level.
It's like counting how many dark pixels vs light pixels you have.

What is Statistical Testing?
---------------------------
Statistical tests help us answer: "Did our image processing actually change the image?"
We use math to measure if the changes are real or just random.

LLMs used: Claude Sonnet 4 (Anthropic)
Author: Qingke He
Date: 2025-08-22
"""

# ============================================================================
# STEP 1: IMPORT THE TOOLS WE NEED
# ============================================================================
# These are like "toolboxes" that give us special functions to work with images

import numpy as np          # For working with numbers and arrays (images are arrays!)
import matplotlib.pyplot as plt  # For creating graphs and showing images
from PIL import Image       # PIL = Python Imaging Library, for loading/saving images
from scipy import stats     # For doing statistical tests
from scipy.ndimage import gaussian_filter  # For making images blurry
from scipy.fft import fft2, ifft2, fftshift, ifftshift  # For fancy math (Fourier transform)
from skimage import util    # For more image processing tools
import os                   # For working with files and folders
import warnings             # For hiding scary warning messages
warnings.filterwarnings('ignore')  # Hide warnings to keep output clean

print("✓ All tools loaded successfully!")


class ImageProcessor:
    """
    This is our main class - think of it as a "machine" that processes images.
    
    A class is like a blueprint for creating objects that can do specific tasks.
    In our case, this ImageProcessor can load images, blur them, and analyze them.
    """
    
    def __init__(self, image_path):
        """
        This is called a "constructor" - it runs when we create a new ImageProcessor.
        
        Think of it like setting up a new machine:
        - We tell it where to find the image
        - We prepare empty storage spaces for our results
        
        Args:
            image_path (str): The file path to our image (like "folder/image.png")
        """
        self.image_path = image_path  # Remember where our image is
        
        # These are like empty boxes where we'll store our results later
        self.original_image = None    # Will store the original image
        self.processed_image = None   # Will store the image after we process it
        self.original_histogram = None  # Will store the original image's histogram
        self.processed_histogram = None # Will store the processed image's histogram
        
        print(f"✓ ImageProcessor created for: {image_path}")
    
    def load_and_check_image(self):
        """
        This function loads our image and makes sure it's the right type.
        
        It's like checking if a key fits a lock before trying to open it.
        
        Returns:
            bool: True if everything worked, False if there was a problem
        """
        try:
            print("🔍 Loading and checking image...")
            
            # Step 1: Open the image file
            # PIL.Image.open() is like opening a book - we can see what's inside
            image = Image.open(self.image_path)
            
            # Step 2: Check if it's a PNG file
            # Different file types work differently, so we want PNG
            if image.format != 'PNG':
                error_msg = f"❌ Wrong file type! We need PNG, but got {image.format}"
                print(error_msg)
                return False
            
            # Step 3: Check the size (dimensions)
            # We need exactly 500x500 pixels
            width, height = image.size
            if width != 500 or height != 500:
                error_msg = f"❌ Wrong size! We need 500x500, but got {width}x{height}"
                print(error_msg)
                return False
            
            # Step 4: Convert to black and white (greyscale)
            # Color images have 3 channels (Red, Green, Blue)
            # Greyscale images have 1 channel (just brightness)
            if image.mode != 'L':  # 'L' means greyscale
                print(f"  🔄 Converting {image.mode} image to greyscale...")
                image = image.convert('L')
            
            # Step 5: Convert to numpy array
            # PIL images are like pictures in a frame
            # Numpy arrays are like spreadsheets of numbers
            # We need numbers to do math on the image
            self.original_image = np.array(image)
            
            # Step 6: Check the data type
            # uint8 means "unsigned 8-bit integer"
            # This means each pixel can be 0-255 (256 different brightness levels)
            if self.original_image.dtype != np.uint8:
                error_msg = f"❌ Wrong data type! We need 8-bit, but got {self.original_image.dtype}"
                print(error_msg)
                return False
            
            # If we get here, everything worked!
            print(f"✅ Image loaded successfully!")
            print(f"   📏 Size: {width}x{height} pixels")
            print(f"   🎨 Type: 8-bit greyscale")
            print(f"   📊 Data range: {self.original_image.min()} to {self.original_image.max()}")
            return True
            
        except Exception as e:
            # If anything goes wrong, this catches the error and shows a friendly message
            print(f"❌ Error loading image: {e}")
            return False
    
    def apply_image_effect(self, effect_type='gaussian', **options):
        """
        This function applies different effects to our image.
        
        Think of it like applying filters in a photo app:
        - Original photo → Apply effect → New photo
        
        Args:
            effect_type (str): What kind of effect we want
            **options: Extra settings for the effect (like how strong it should be)
        """
        # First, make sure we have an image to work with
        if self.original_image is None:
            print("❌ No image loaded! Call load_and_check_image() first.")
            return
        
        print(f"🎨 Applying {effect_type} effect...")
        
        if effect_type == 'gaussian':
            # Gaussian blur is like looking through frosted glass
            # It makes everything smoothly blurry
            blur_strength = options.get('sigma', 2.0)  # Default blur strength
            self.processed_image = gaussian_filter(self.original_image, sigma=blur_strength)
            print(f"  ✅ Applied Gaussian blur (strength: {blur_strength})")
            
        elif effect_type == 'fourier':
            # Fourier transform is fancy math that works in "frequency space"
            # High frequencies = fine details (like edges)
            # Low frequencies = big shapes (like backgrounds)
            # We remove high frequencies to blur the image
            cutoff = options.get('cutoff_freq', 0.1)  # How much detail to remove
            self.processed_image = self._apply_fourier_filter(cutoff)
            print(f"  ✅ Applied Fourier filter (removed {cutoff*100}% of details)")
            
        elif effect_type == 'average':
            # Average filter is like smearing paint with your finger
            # Each pixel becomes the average of itself and its neighbors
            kernel_size = options.get('kernel_size', 5)  # How many neighbors to average
            self.processed_image = self._apply_average_filter(kernel_size)
            print(f"  ✅ Applied average filter (averaging {kernel_size}x{kernel_size} pixels)")
            
        elif effect_type == 'mosaic':
            # Mosaic effect is like looking through a window with many small panes
            # Each pane shows the average color of that area
            tile_size = options.get('tile_size', 20)  # Size of each "pane"
            self.processed_image = self._apply_mosaic_effect(tile_size)
            print(f"  ✅ Applied mosaic effect (tile size: {tile_size}x{tile_size})")
            
        else:
            print(f"❌ Unknown effect type: {effect_type}")
            print("   Available effects: gaussian, fourier, average, mosaic")
            return
    
    def _apply_fourier_filter(self, cutoff_freq):
        """
        This is a helper function that does the Fourier transform math.
        
        Don't worry if you don't understand the math - just know that:
        1. We convert the image to "frequency space"
        2. We remove the high frequencies (details)
        3. We convert it back to normal image space
        
        Args:
            cutoff_freq (float): How much detail to remove (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: The blurred image
        """
        # Get image dimensions
        rows, cols = self.original_image.shape
        
        # Step 1: Convert to frequency space using FFT (Fast Fourier Transform)
        # This is like translating the image into a different language
        f_transform = fft2(self.original_image)
        
        # Step 2: Shift the frequencies so low frequencies are in the center
        f_shift = fftshift(f_transform)
        
        # Step 3: Find the center of the image
        center_row, center_col = rows // 2, cols // 2
        
        # Step 4: Create a circular mask
        # This mask keeps low frequencies (center) and removes high frequencies (edges)
        y, x = np.ogrid[:rows, :cols]
        # Calculate distance from center for each pixel
        distance_from_center = (x - center_col)**2 + (y - center_row)**2
        # Create mask: True for pixels we keep, False for pixels we remove
        mask = distance_from_center <= (cutoff_freq * min(rows, cols))**2
        
        # Step 5: Apply the mask (remove high frequencies)
        f_shift_filtered = f_shift * mask
        
        # Step 6: Convert back to normal image space
        f_ishift = ifftshift(f_shift_filtered)
        img_back = ifft2(f_ishift)
        
        # Step 7: Convert to real numbers and proper format
        return np.abs(img_back).astype(np.uint8)
    
    def _apply_average_filter(self, kernel_size):
        """
        This function applies a simple averaging filter.
        
        Think of it like this:
        1. Look at a pixel and its neighbors
        2. Calculate the average brightness of all of them
        3. Make the center pixel that average brightness
        4. Repeat for every pixel
        
        Args:
            kernel_size (int): How many neighbors to include (must be odd number)
            
        Returns:
            numpy.ndarray: The blurred image
        """
        # Create a "kernel" - this is like a template that slides over the image
        # The kernel is a square of 1's, divided by the total number of pixels
        # This makes the average calculation work
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        # Apply the kernel to the image using convolution
        # Convolution is like sliding the kernel over every pixel
        return util.convolve(self.original_image, kernel, mode='reflect')
    
    def _apply_mosaic_effect(self, tile_size):
        """
        This function creates a mosaic effect by dividing the image into tiles.
        
        How it works:
        1. Divide the image into squares (tiles)
        2. For each tile, calculate the average color
        3. Make every pixel in that tile the same average color
        
        Args:
            tile_size (int): Size of each tile in pixels
            
        Returns:
            numpy.ndarray: The mosaic image
        """
        # Get image dimensions
        height, width = self.original_image.shape
        
        # Create a copy of the original image to work with
        mosaic_image = np.copy(self.original_image)
        
        # Step through the image tile by tile
        # range(0, height, tile_size) means: start at 0, go to height, step by tile_size
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                
                # Calculate the boundaries of this tile
                # Make sure we don't go past the edge of the image
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                
                # Extract this tile from the image
                tile = self.original_image[y:y_end, x:x_end]
                
                # Calculate the average brightness of all pixels in this tile
                tile_average = np.mean(tile)
                
                # Set every pixel in this tile to the average value
                mosaic_image[y:y_end, x:x_end] = tile_average
        
        # Convert back to the right data type
        return mosaic_image.astype(np.uint8)
    
    def create_histograms(self):
        """
        This function creates histograms to show how the image processing changed the pixel distribution.
        
        A histogram is like a bar chart showing:
        - X-axis: Brightness levels (0 = black, 255 = white)
        - Y-axis: How many pixels have each brightness level
        
        This helps us see if our processing made the image darker, lighter, or changed the contrast.
        """
        # Make sure we have both images to compare
        if self.original_image is None or self.processed_image is None:
            print("❌ Need both original and processed images!")
            return
        
        print("📊 Creating histograms...")
        
        # Calculate histograms for both images
        # np.histogram counts how many pixels fall into each brightness bin
        # bins=256 means we have 256 different brightness levels (0-255)
        # range=[0, 256] means we're looking at brightness values from 0 to 255
        self.original_histogram, bins = np.histogram(self.original_image.flatten(), bins=256, range=[0, 256])
        self.processed_histogram, _ = np.histogram(self.processed_image.flatten(), bins=256, range=[0, 256])
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Histogram comparison
        ax1.plot(bins[:-1], self.original_histogram, 'blue', alpha=0.7, label='Original Image', linewidth=2)
        ax1.plot(bins[:-1], self.processed_histogram, 'red', alpha=0.7, label='Processed Image', linewidth=2)
        ax1.set_xlabel('Pixel Brightness (0 = Black, 255 = White)')
        ax1.set_ylabel('Number of Pixels')
        ax1.set_title('How Did Our Processing Change the Image?')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 255)
        
        # Right plot: Image comparison
        # np.hstack combines the two images side by side
        combined_images = np.hstack([self.original_image, self.processed_image])
        ax2.imshow(combined_images, cmap='gray', aspect='auto')
        ax2.set_title('Original Image | Processed Image')
        ax2.axis('off')
        
        # Make the plot look nice
        plt.suptitle('Histogram Analysis: Before vs After Processing', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        print("✅ Histograms created and displayed!")
    
    def run_statistical_tests(self):
        """
        This function runs statistical tests to see if our image processing actually made a difference.
        
        Statistical tests help us answer: "Are the changes real, or just random noise?"
        
        We use three different tests because each one looks at the data differently:
        1. Chi-square: Tests overall distribution differences
        2. Kolmogorov-Smirnov: Tests shape differences  
        3. Mann-Whitney: Tests central tendency differences
        
        Returns:
            dict: Results of all statistical tests
        """
        # Make sure we have histograms to test
        if self.original_histogram is None or self.processed_histogram is None:
            print("❌ Need histograms first! Call create_histograms() first.")
            return None
        
        print("\n" + "="*60)
        print("🔬 STATISTICAL ANALYSIS RESULTS")
        print("="*60)
        print("Testing if our image processing made a real difference...")
        print("(P-values < 0.05 mean the difference is statistically significant)")
        print()
        
        # Store all our test results
        test_results = {}
        
        # Test 1: Chi-square test
        # This test compares the overall distribution of pixel values
        try:
            chi2_stat, chi2_p = stats.chisquare(self.original_histogram, self.processed_histogram)
            test_results['chi2'] = (chi2_stat, chi2_p)
            
            print("📊 Test 1: Chi-square Test")
            print("   What it tests: Overall distribution differences")
            print(f"   Test statistic: {chi2_stat:.4f}")
            print(f"   P-value: {chi2_p:.6f}")
            
            if chi2_p < 0.05:
                print("   ✅ Result: SIGNIFICANT difference detected!")
            else:
                print("   ❌ Result: No significant difference detected")
            print()
            
        except Exception as e:
            print(f"📊 Test 1: Chi-square Test - FAILED ({e})")
            test_results['chi2'] = (None, None)
            print()
        
        # Test 2: Kolmogorov-Smirnov test
        # This test compares the shape of the distributions
        try:
            ks_stat, ks_p = stats.ks_2samp(self.original_histogram, self.processed_histogram)
            test_results['ks'] = (ks_stat, ks_p)
            
            print("📊 Test 2: Kolmogorov-Smirnov Test")
            print("   What it tests: Shape and spread differences")
            print(f"   Test statistic: {ks_stat:.4f}")
            print(f"   P-value: {ks_p:.6f}")
            
            if ks_p < 0.05:
                print("   ✅ Result: SIGNIFICANT difference detected!")
            else:
                print("   ❌ Result: No significant difference detected")
            print()
            
        except Exception as e:
            print(f"📊 Test 2: Kolmogorov-Smirnov Test - FAILED ({e})")
            test_results['ks'] = (None, None)
            print()
        
        # Test 3: Mann-Whitney U test
        # This test compares the central tendency (like median) of the distributions
        try:
            mw_stat, mw_p = stats.mannwhitneyu(self.original_histogram, self.processed_histogram, 
                                              alternative='two-sided')
            test_results['mw'] = (mw_stat, mw_p)
            
            print("📊 Test 3: Mann-Whitney U Test")
            print("   What it tests: Central tendency differences")
            print(f"   Test statistic: {mw_stat:.4f}")
            print(f"   P-value: {mw_p:.6f}")
            
            if mw_p < 0.05:
                print("   ✅ Result: SIGNIFICANT difference detected!")
            else:
                print("   ❌ Result: No significant difference detected")
            print()
            
        except Exception as e:
            print(f"📊 Test 3: Mann-Whitney U Test - FAILED ({e})")
            test_results['mw'] = (None, None)
            print()
        
        # Summary
        significant_tests = 0
        total_tests = 0
        
        for test_name, (stat, p_val) in test_results.items():
            if p_val is not None:
                total_tests += 1
                if p_val < 0.05:
                    significant_tests += 1
        
        print("📋 SUMMARY")
        print(f"   Tests run: {total_tests}")
        print(f"   Significant differences found: {significant_tests}")
        print(f"   Conclusion: ", end="")
        
        if significant_tests == 0:
            print("Our processing didn't make a significant change to the image.")
        elif significant_tests == total_tests:
            print("Our processing made a very significant change to the image!")
        else:
            print(f"Our processing made some significant changes to the image ({significant_tests}/{total_tests} tests).")
        
        return test_results
    
    def save_processed_images(self, output_folder="data/output"):
        """
        This function saves our processed images to files.
        
        We save two versions:
        1. Full-size processed image (500x500)
        2. Smaller version (250x250) for easier viewing
        
        Args:
            output_folder (str): Where to save the images
        """
        if self.processed_image is None:
            print("❌ No processed image to save! Apply an effect first.")
            return
        
        print("💾 Saving processed images...")
        
        # Create the output folder if it doesn't exist
        # os.makedirs() is like creating a new folder on your computer
        os.makedirs(output_folder, exist_ok=True)
        
        # Save the full-size processed image
        full_size_path = os.path.join(output_folder, "processed_image.png")
        # Convert numpy array back to PIL Image and save
        Image.fromarray(self.processed_image).save(full_size_path)
        print(f"  ✅ Full-size image saved: {full_size_path}")
        
        # Create and save a smaller version
        # This is useful for sharing or viewing on smaller screens
        smaller_image = Image.fromarray(self.processed_image).resize((250, 250), Image.Resampling.LANCZOS)
        smaller_path = os.path.join(output_folder, "processed_image_250x250.png")
        smaller_image.save(smaller_path)
        print(f"  ✅ Smaller version saved: {smaller_path}")
        
        print(f"📁 Images saved in: {os.path.abspath(output_folder)}")
    
    def run_complete_analysis(self, effect_type='gaussian', **options):
        """
        This function runs the complete image processing pipeline.
        
        It's like a recipe that does everything step by step:
        1. Load the image
        2. Apply an effect
        3. Create histograms
        4. Run statistical tests
        5. Save the results
        
        Args:
            effect_type (str): What effect to apply
            **options: Settings for the effect
        """
        print("🚀 Starting complete image analysis pipeline...")
        print("="*60)
        
        # Step 1: Load and check the image
        if not self.load_and_check_image():
            print("❌ Failed to load image. Stopping.")
            return False
        
        # Step 2: Apply the chosen effect
        self.apply_image_effect(effect_type, **options)
        
        # Step 3: Create histograms to visualize the changes
        self.create_histograms()
        
        # Step 4: Run statistical tests to measure the changes
        test_results = self.run_statistical_tests()
        
        # Step 5: Save the processed images
        self.save_processed_images()
        
        print("\n" + "="*60)
        print("🎉 Pipeline completed successfully!")
        print("="*60)
        
        return True


def main():
    """
    This is the main function that runs when you execute the program.
    
    It demonstrates all the different effects you can apply to images.
    Think of it as a "demo mode" that shows what the program can do.
    """
    print("🎨 IMAGE PROCESSING PROGRAM FOR BEGINNERS")
    print("="*50)
    print("This program will show you different ways to process images.")
    print("Each method will create a different effect on your image.")
    print()
    
    # Create our image processor
    # We'll use the bootcamp homework image
    processor = ImageProcessor("data/input/bootcamp_homework_500x500.png")
    
    # Define all the effects we want to try
    # Each effect has different parameters you can adjust
    effects_to_try = [
        ('gaussian', {'sigma': 2.0}, 'Gaussian Blur (smooth blur)'),
        ('fourier', {'cutoff_freq': 0.1}, 'Fourier Filter (frequency-based blur)'),
        ('average', {'kernel_size': 5}, 'Average Filter (simple blur)'),
        ('mosaic', {'tile_size': 20}, 'Mosaic Effect (pixelated tiles)')
    ]
    
    # Try each effect one by one
    for effect_name, effect_params, effect_description in effects_to_try:
        print(f"\n{'='*20} {effect_name.upper()} {'='*20}")
        print(f"Effect: {effect_description}")
        print(f"Parameters: {effect_params}")
        print()
        
        # Create a new processor for each effect (to avoid mixing results)
        processor = ImageProcessor("data/input/bootcamp_homework_500x500.png")
        
        # Run the complete analysis with this effect
        success = processor.run_complete_analysis(effect_name, **effect_params)
        
        if success:
            print(f"✅ {effect_name} analysis completed successfully!")
        else:
            print(f"❌ {effect_name} analysis failed!")
        
        print("\n" + "-"*50)


if __name__ == "__main__":
    """
    This special line means: "Only run the main function if this file is run directly"
    
    In other words:
    - If you run: python image_processor.py → main() runs
    - If you import this file in another program → main() doesn't run
    
    This is a common Python pattern that makes the code more flexible.
    """
    main()
