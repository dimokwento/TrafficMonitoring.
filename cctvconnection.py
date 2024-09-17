from dvrip import DVRIPCam
from time import sleep

host_ip = '192.168.68.73'

cam = DVRIPCam(host_ip, user='Office_Sample', password='')
if cam.login():
	print("Success! Connected to " + host_ip)
else:
	print("Failure. Could not connect.")

print("Camera time:", cam.get_time())

# Reboot camera
cam.reboot()
sleep(60) # wait while camera starts

# Login again
cam.login()
# Sync camera time with PC time
cam.set_time()
# Disconnect
cam.close()