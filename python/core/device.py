class Device:
    def __init__(self, device_type, device_id=0):
        self.device_type = device_type  # "cpu" or "cuda"
        self.device_id = device_id
        
    @staticmethod
    def cpu():
        return Device("cpu", 0)
    
    @staticmethod
    def cuda(device_id=0):
        return Device("cuda", device_id)
    
    def __repr__(self):
        if self.device_type == "cpu":
            return "cpu()"
        return f"cuda({self.device_id})"

cpu = Device.cpu()
cuda = Device.cuda