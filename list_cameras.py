from vmbpy import *

# function to list cameras (specifically Allied Vision Industrial cameras) 

def list_cameras():
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print('No Cameras accessible.')
        else:
            for cam in cams:
                print(f"Camera ID: {cam.get_id()}, Model: {cam.get_model()}, Name: {cam.get_name()}")

if __name__ == "__main__":
    list_cameras()
