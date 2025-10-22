# Noisy Reducer AI

This project implements a noise reduction AI model that processes audio files to reduce background noise. It utilizes deep learning techniques to enhance audio quality, making it suitable for various applications in audio processing. The project includes a Jupyter notebook for predictions, sample audio files for testing, and necessary scripts for model handling.

## Project Structure

- **src/make_predict.ipynb**: Jupyter notebook containing the main code for predicting and reducing noise from audio files using a trained model.
- **src/scripts/nocle.py**: Python script with utility functions or classes used in the noise reduction process.
- **src/model/first.hdf5**: Trained Keras model saved in HDF5 format for audio noise reduction.
- **src/model/quantized_model.tflite**: Quantized version of the model, optimized for mobile and edge devices.
- **test/noisy_testset_wav/1.wav**: Sample noisy audio file used for testing the noise reduction model.
- **test/clean_testset_wav/1.wav**: Sample clean audio file used for comparison with the denoised output.
- **requirements.txt**: Lists the Python dependencies required to run the project.
- **.gitignore**: Specifies files and directories that should be ignored by Git.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

Open the Jupyter notebook `make_predict.ipynb` to start using the noise reduction model. You can load your own audio files and see the results of the noise reduction process.

## License

This project is licensed under the MIT License.