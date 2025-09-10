# Image Processing and Analysis Project

This project implements image processing techniques including image blurring, histogram analysis, and statistical testing to detect differences between original and processed images.

## Project Overview

The project successfully demonstrates:
- Loading and validating 500x500 PNG greyscale images
- Applying multiple blurring methods (Gaussian, Fourier Low-Pass, Average)
- Calculating and comparing greyscale histograms
- Performing statistical analysis using multiple tests
- Saving processed images and downscaled versions

## Features

### Blurring Methods
1. **Gaussian Filter**: Smooth blur using Gaussian kernel with configurable sigma
2. **Fourier Low-Pass Filter**: Frequency domain filtering with adjustable cutoff frequency
3. **Average Filter**: Simple spatial domain averaging with configurable kernel size

### Statistical Analysis
- **Chi-square test**: Tests for differences in distribution
- **Kolmogorov-Smirnov test**: Tests for differences in cumulative distribution  
- **Mann-Whitney U test**: Non-parametric test for differences in central tendency

### Output Generation
- Blurred images saved as PNG files
- 250x250 pixel downscaled versions
- Comprehensive histogram plots and statistical results

## Requirements

### Software Dependencies
```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
Pillow>=8.3.0
scikit-image>=0.18.0
```

### Installation
```bash
# Clone the repository
git clone [your-repo-url]
cd image-processing-project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from image_processor import ImageProcessor

# Initialize processor with image path
processor = ImageProcessor("data/input/grandma_500x500.png")

# Run complete pipeline with Gaussian blur
processor.process_pipeline('gaussian', sigma=2.0)
```

### Advanced Usage
```python
# Load and validate image
processor.load_and_validate_image()

# Apply specific blur method
processor.apply_blur('fourier', cutoff_freq=0.1)

# Plot histograms
processor.plot_histograms()

# Perform statistical analysis
stats_results = processor.statistical_analysis()

# Save outputs
processor.save_outputs()
```

### Command Line Usage
```bash
# Run with default settings
python image_processor.py

# Run with specific blur method
python -c "
from image_processor import ImageProcessor
p = ImageProcessor('data/input/grandma_500x500.png')
p.process_pipeline('average', kernel_size=7)
"
```

## Project Structure
```
image-processing-project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── image_processor.py       # Main processing module
├── data/
│   ├── input/               # Input images
│   │   └── grandma_500x500.png
│   └── output/              # Processed images
│       ├── *_blurred.png    # Blurred versions
│       └── *_downscaled_250x250.png  # Downscaled versions
└── tests/                    # Unit tests (if added)
```

## Input Image Requirements

- **Format**: PNG
- **Dimensions**: 500x500 pixels
- **Color**: 8-bit greyscale
- **File size**: Variable (typically < 1MB)

## Output Files

The program generates several output files:
1. **Blurred images**: Full-size blurred versions using different methods
2. **Downscaled images**: 250x250 pixel versions of blurred images
3. **Console output**: Statistical analysis results and processing status
4. **Visual plots**: Histogram comparisons displayed during execution

## Statistical Analysis

The program performs three statistical tests to detect differences:

1. **Chi-square test**: Detects distribution differences
2. **Kolmogorov-Smirnov test**: Detects shape differences  
3. **Mann-Whitney U test**: Detects central tendency differences

Results include p-values and significance indicators (α=0.05).

## Technical Details

### Image Processing Algorithms
- **Gaussian Filter**: Uses scipy.ndimage.gaussian_filter with configurable sigma
- **Fourier Filter**: Implements FFT-based low-pass filtering with circular mask
- **Average Filter**: Applies uniform kernel convolution with boundary reflection

### Performance Considerations
- Efficient numpy array operations
- Optimized FFT implementations
- Memory-efficient histogram calculations
- Configurable parameters for different use cases

## Error Handling

The program includes comprehensive error handling for:
- Invalid image formats
- Incorrect image dimensions
- Missing input files
- Processing failures
- Output directory creation

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

## Author

**Qingke He**  
Date: 2025-08-22

## Acknowledgments

- LLM assistance: Claude Sonnet 4 (Anthropic)
- Scientific computing libraries: NumPy, SciPy, scikit-image
- Image processing: Pillow (PIL)
- Visualization: Matplotlib

## Version History

- **v1.0.0** (2025-08-22): Initial release with core functionality
  - Image loading and validation
  - Multiple blurring methods
  - Histogram analysis and plotting
  - Statistical testing
  - Output generation
