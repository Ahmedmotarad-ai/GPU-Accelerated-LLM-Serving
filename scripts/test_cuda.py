import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
    print("cuda_version", torch.version.cuda)
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print("matmul_ok", round(y, 3))
else:
    print("cuda not available")