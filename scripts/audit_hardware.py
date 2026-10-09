import ctypes
import os

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

stat = MEMORYSTATUSEX()
stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

print("=== EXACT HARDWARE AUDIT ===")
print(f"Total Physical RAM:     {stat.ullTotalPhys / (1024**3):.2f} GB")
print(f"Available Physical RAM: {stat.ullAvailPhys / (1024**3):.2f} GB")
print(f"RAM Utilization:        {stat.dwMemoryLoad}%")
print(f"Logical CPU Threads:    {os.cpu_count()}")
