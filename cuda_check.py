import torch
print(f'Available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
print(f'Cuda version: {torch.version.cuda}')