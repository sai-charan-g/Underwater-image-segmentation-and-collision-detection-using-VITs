@echo off
echo ================================================
echo  Underwater Segmentation - Local GPU Setup
echo ================================================

echo.
echo [1/3] Installing CUDA PyTorch (cu121)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/3] Installing project dependencies...
pip install -r requirements.txt

echo.
echo [3/3] Verifying GPU...
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ================================================
echo  Setup complete! Run with: python train.py
echo ================================================
pause
