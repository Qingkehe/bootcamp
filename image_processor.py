#!/usr/bin/env python3
"""
Image Processing Module for Image Blurring and Analysis

This module provides functionality to:
- Load and validate PNG images
- Apply various blurring techniques
- Calculate and compare histograms
- Perform statistical analysis
- Save processed images

LLMs used: Claude Sonnet 4 (Anthropic)

Author: [Qingke He]
Date: [2025-08-22]
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import util
import os
import warnings
warnings.filterwarnings('ignore')


class ImageProcessor:
    """
    A class to process images with various blurring techniques and analysis.
    """
    
    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with an image path.
        
        Args:
            image_path (str): Path to the input image file
        """
        self.image_path = image_path
        self.original_image = None
        self.blurred_image = None
        self.original_hist = None
        self.blurred_hist = None
        
    def load_and_validate_image(self):
        """
        Load and validate the input image.
        
        Returns:
            bool: True if image is valid, False otherwise
            
        Raises:
            ValueError: If image doesn't meet requirements
        """
        try:
            # Load image
            self.original_image = Image.open(self.image_path)
            
            # Check if image is PNG
            if self.original_image.format != 'PNG':
                raise ValueError("Image must be in PNG format")
            
            # Check dimensions
            width, height = self.original_image.size
            if width != 500 or height != 500:
                raise ValueError(f"Image must be 500x500 pixels, got {width}x{height}")
            
            # Convert to greyscale if not already
            if self.original_image.mode != 'L':
                self.original_image = self.original_image.convert('L')
            
            # Convert to numpy array
            self.original_image = np.array(self.original_image)
            
            # Check bit depth (should be 8-bit)
            if self.original_image.dtype != np.uint8:
                raise ValueError("Image must be 8-bit")
            
            print(f"✓ Image loaded successfully: {width}x{height} pixels, 8-bit greyscale")
            return True
            
        except Exception as e:
            print(f"✗ Error loading image: {e}")
            return False
    
    def apply_blur(self, method='gaussian', **kwargs):
        """
        Apply blurring to the image using specified method.
        
        Args:
            method (str): Blur method ('gaussian', 'fourier', 'average')
            **kwargs: Additional parameters for specific methods
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Call load_and_validate_image() first.")
        
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 2.0)
            self.blurred_image = gaussian_filter(self.original_image, sigma=sigma)
            print(f"✓ Applied Gaussian blur with sigma={sigma}")
            
        elif method == 'fourier':
            cutoff_freq = kwargs.get('cutoff_freq', 0.1)
            self.blurred_image = self._fourier_low_pass_filter(cutoff_freq)
            print(f"✓ Applied Fourier low-pass filter with cutoff={cutoff_freq}")
            
        elif method == 'average':
            kernel_size = kwargs.get('kernel_size', 5)
            self.blurred_image = self._average_filter(kernel_size)
            print(f"✓ Applied average filter with kernel size={kernel_size}")
            
        else:
            raise ValueError(f"Unknown blur method: {method}")
    
    def _fourier_low_pass_filter(self, cutoff_freq):
        """
        Apply Fourier low-pass filter for blurring.
        
        Args:
            cutoff_freq (float): Cutoff frequency (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: Blurred image
        """
        # Apply FFT
        f_transform = fft2(self.original_image)
        f_shift = fftshift(f_transform)
        
        # Create low-pass filter
        rows, cols = self.original_image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create circular mask
        y, x = np.ogrid[:rows, :cols]
        mask = (x - ccol)**2 + (y - crow)**2 <= (cutoff_freq * min(rows, cols))**2
        
        # Apply filter
        f_shift_filtered = f_shift * mask
        
        # Inverse FFT
        f_ishift = ifftshift(f_shift_filtered)
        img_back = ifft2(f_ishift)
        
        return np.abs(img_back).astype(np.uint8)
    
    def _average_filter(self, kernel_size):
        """
        Apply simple average filter for blurring.
        
        Args:
            kernel_size (int): Size of the averaging kernel
            
        Returns:
            numpy.ndarray: Blurred image
        """
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return util.convolve(self.original_image, kernel, mode='reflect')
    
    def calculate_histogram(self, image):
        """
        Calculate greyscale histogram for an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (histogram values, bin edges)
        """
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
        return hist, bins
    
    def plot_histograms(self):
        """
        Plot histograms of original and blurred images.
        """
        if self.original_image is None or self.blurred_image is None:
            raise ValueError("Both original and blurred images must be available")
        
        # Calculate histograms
        self.original_hist, bins = self.calculate_histogram(self.original_image)
        self.blurred_hist, _ = self.calculate_histogram(self.blurred_image)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(bins[:-1], self.original_hist, 'b-', alpha=0.7, label='Original Image')
        plt.plot(bins[:-1], self.blurred_hist, 'r-', alpha=0.7, label='Blurred Image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Greyscale Histogram Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.hstack([self.original_image, self.blurred_image]), 
                  cmap='gray', aspect='auto')
        plt.title('Original | Blurred')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Histograms plotted successfully")
    
    def statistical_analysis(self):
        """
        Perform statistical analysis on the histograms.
        """
        if self.original_hist is None or self.blurred_hist is None:
            raise ValueError("Histograms not calculated. Call plot_histograms() first.")
        
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS RESULTS")
        print("="*50)
        
        # Chi-square test for distribution differences
        chi2_stat, chi2_p = stats.chisquare(self.original_hist, self.blurred_hist)
        print(f"Chi-square test:")
        print(f"  Statistic: {chi2_stat:.4f}")
        print(f"  P-value: {chi2_p:.6f}")
        print(f"  Significant difference: {'Yes' if chi2_p < 0.05 else 'No'} (α=0.05)")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(self.original_hist, self.blurred_hist)
        print(f"\nKolmogorov-Smirnov test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_p:.6f}")
        print(f"  Significant difference: {'Yes' if ks_p < 0.05 else 'No'} (α=0.05)")
        
        # Mann-Whitney U test
        mw_stat, mw_p = stats.mannwhitneyu(self.original_hist, self.blurred_hist, 
                                          alternative='two-sided')
        print(f"\nMann-Whitney U test:")
        print(f"  Statistic: {mw_stat:.4f}")
        print(f"  P-value: {mw_p:.6f}")
        print(f"  Significant difference: {'Yes' if mw_p < 0.05 else 'No'} (α=0.05)")
        
        # Summary
        significant_tests = sum([chi2_p < 0.05, ks_p < 0.05, mw_p < 0.05])
        print(f"\nSummary: {significant_tests}/3 tests show significant differences")
        
        return {
            'chi2': (chi2_stat, chi2_p),
            'ks': (ks_stat, ks_p),
            'mw': (mw_stat, mw_p)
        }
    
    def save_outputs(self, output_dir="data/output"):
        """
        Save processed images to output directory.
        
        Args:
            output_dir (str): Output directory path
        """
        if self.blurred_image is None:
            raise ValueError("No blurred image available. Apply blur first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save blurred image
        blurred_path = os.path.join(output_dir, "blurred_image.png")
        Image.fromarray(self.blurred_image).save(blurred_path)
        print(f"✓ Blurred image saved: {blurred_path}")
        
        # Create and save downscaled image (250x250)
        downscaled = Image.fromarray(self.blurred_image).resize((250, 250), Image.Resampling.LANCZOS)
        downscaled_path = os.path.join(output_dir, "downscaled_250x250.png")
        downscaled.save(downscaled_path)
        print(f"✓ Downscaled image saved: {downscaled_path}")
        
        return blurred_path, downscaled_path
    
    def process_pipeline(self, blur_method='gaussian', **kwargs):
        """
        Run the complete image processing pipeline.
        
        Args:
            blur_method (str): Method for blurring
            **kwargs: Parameters for blurring method
        """
        print("Starting image processing pipeline...")
        print("="*50)
        
        # Step 1: Load and validate
        if not self.load_and_validate_image():
            return False
        
        # Step 2: Apply blur
        self.apply_blur(blur_method, **kwargs)
        
        # Step 3: Plot histograms
        self.plot_histograms()
        
        # Step 4: Statistical analysis
        stats_results = self.statistical_analysis()
        
        # Step 5: Save outputs
        self.save_outputs()
        
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        print("="*50)
        
        return True


def main():
    """
    Main function to demonstrate the ImageProcessor class.
    """
    # Example usage
    processor = ImageProcessor("data/input/grandma_500x500.png")
    
    # Run with different blur methods
    methods = [
        ('gaussian', {'sigma': 2.0}),
        ('fourier', {'cutoff_freq': 0.1}),
        ('average', {'kernel_size': 5})
    ]
    
    for method, params in methods:
        print(f"\n{'='*20} {method.upper()} BLUR {'='*20}")
        processor = ImageProcessor("data/input/grandma_500x500.png")
        processor.process_pipeline(method, **params)


if __name__ == "__main__":
    main()
