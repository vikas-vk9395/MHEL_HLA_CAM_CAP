
from tendo import singleton
me = singleton.SingleInstance()


import os
import cv2
import time
import logging
import traceback
import datetime
import pymysql
from pypylon import pylon
from threading import Thread
from ipaddress import ip_address

import snap7
from snap7 import util

# Constants
WRITE_ON = 1
WRITE_OFF = 0
POSITION_OFFSET = 2
SLEEP_INTERVAL = 1
POSITION_WRITE_PLC = "D105703"
# MODEL = "D105751"

ENGINE_NO = "D105751"

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "insightzz@123"
DB_NAME = "hla_cam_cap_db"

CAM1 ='24117731'

# Set up logger
logger = logging.getLogger("MAHINDRA_CAM_CAP_HLA_FRAME_CAPTURE")
logger.setLevel(logging.DEBUG)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_fh = logging.FileHandler("MAHINDRA_CAM_CAP_HLA_FRAME_CAPTURE.log", mode='a')
logger_fh.setFormatter(log_format)
logger.addHandler(logger_fh)


class PLC_Write_Trigger_Value:
    PLC_TRIGGER_INSP_OK = 1
    PLC_TRIGGER_INSP_NOT_OK = 2
    capture_bit_send_plc = 1
 
class PLC_READ_DB_COLUMN_BUFFER_POSITION:
    engineNumberStringBufferPosition = 0
    COL_CYCLE_START_INT_start_buffer = 512
    result_write_buffer_offset=14
    write_capture_bit_to_plc = 4

engineNumberStringBufferPosition = 0
COL_CYCLE_START_INT_start_buffer = 512

SAVE_IMAGE_DIR_CAM_CAP_HLA = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/FRAME_CAPTURE/IMG/"


def update_PLC_Status_UI(STATUS):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        query = "UPDATE status_config_table SET PLC_STATUS = %s WHERE ID = 1"
        cur.execute(query, (STATUS))
        db_update.commit()
    except Exception as e:
        print("update_PLC_Status_UI() Exception is : "+ str(e))
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close()

def update_PLC_postion_Status(STATUS):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        query = "UPDATE status_config_table SET POSITION = %s WHERE ID = 1"
        cur.execute(query, (STATUS))
        db_update.commit()
    except Exception as e:
        print("update_PLC_postion_Status() Exception is : "+ str(e))
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close()

def update_Camera_Status_UI(STATUS):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        query = "UPDATE status_config_table SET CAMERA_STATUS = %s WHERE ID = 1"
        cur.execute(query, (STATUS))
        db_update.commit()
    except Exception as e:
        print("update_Camera_Status_UI() Exception is : "+ str(e))
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close()

class CameraCapture:
    def __init__(self, DB_HOST, DB_USER, DB_PASS, DB_NAME, CAM1, logger=None):
        self.DB_HOST = DB_HOST
        self.DB_USER = DB_USER
        self.DB_PASS = DB_PASS
        self.DB_NAME = DB_NAME
        self.CAM1 = CAM1
        self.logger = logger if logger else self.setup_logger("CameraCapture")

    def setup_logger(self, file_name):
        logger = logging.getLogger(file_name)
        logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger_fh = logging.FileHandler(f"{file_name}.log", mode='a')
        logger_fh.setFormatter(log_format)
        logger_fh.setLevel(logging.DEBUG)
        logger.addHandler(logger_fh)
        return logger

    def connect_to_camera(self, serial_number):
        cam_list = pylon.TlFactory.GetInstance().EnumerateDevices()
        for cam in cam_list:
            if cam.GetSerialNumber() == serial_number:
                return pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cam))
        return None

    def capture_and_save_image(self, camera,SAVE_IMAGE_DIR_CAM_CAP_HLA, Engine_No,imagcount):
        try:
            grab = None
            if camera.IsGrabbing():
                grab = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

            if grab and grab.GrabSucceeded():
                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

                # Convert to OpenCV BGR8 format
                image = converter.Convert(grab)
                img = image.GetArray()

                # Ensure the directory exists
                if not os.path.exists(SAVE_IMAGE_DIR_CAM_CAP_HLA):
                    os.makedirs(SAVE_IMAGE_DIR_CAM_CAP_HLA)

                TodaysDate = datetime.datetime.now().strftime('%Y_%m_%d')
                if os.path.exists(os.path.join(SAVE_IMAGE_DIR_CAM_CAP_HLA, TodaysDate)) is False: 
                    os.mkdir(os.path.join(SAVE_IMAGE_DIR_CAM_CAP_HLA, TodaysDate))
                if os.path.exists(os.path.join(os.path.join(SAVE_IMAGE_DIR_CAM_CAP_HLA, TodaysDate),Engine_No)) is False:
                    os.mkdir(os.path.join(os.path.join(SAVE_IMAGE_DIR_CAM_CAP_HLA, TodaysDate),Engine_No))
                image_folder_link1 = os.path.join(SAVE_IMAGE_DIR_CAM_CAP_HLA, TodaysDate)
                image_folder_link=(image_folder_link1+"/"+Engine_No)+"/"
                
                current_datetime = datetime.datetime.now()
                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                # Save image with timestamp and engine number
                image_filename = f"POS_{imagcount}.jpg"
                image_path = os.path.join(image_folder_link, image_filename)
                imgathrane_file = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/ALGORITHAM/IMG/"
                image_path_new = os.path.join(imgathrane_file, image_filename)
                img = cv2.rotate(img, cv2.ROTATE_180)
                cv2.imwrite(image_path, img)
                time.sleep(0.02)
                cv2.imwrite(image_path_new, img)
                print(f"Image captured and saved: {image_path}")
                update_Camera_Status_UI("OK")
                # Release the grab result after use
                grab.Release()
            else:
                update_Camera_Status_UI("NOT OK")
                print("Failed to grab an image.")
                self.logger.debug("Failed to grab an image.")
        except Exception as e:
            print(e)
            update_Camera_Status_UI("NOT OK")
            self.logger.debug(f"Exception in capture_and_save_image: {str(e)}")
    
    def set_camera_node(self,camera, node_name, value):
        try:
            if camera.GetNodeMap().GetNode(node_name) is not None:
                node = camera.GetNodeMap().GetNode(node_name)
                node.SetValue(value)
                print(f"{node_name} set to {value}")
            else:
                print(f"Node {node_name} does not exist on this camera.")
        except Exception as e:
            print(f"Error setting {node_name}: {str(e)}")

    def start_capture(self, Engine_No, CAPTURE_FLAG,imagcount):
        try:
            SAVE_IMAGE_DIR_CAM_CAP_HLA = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/FRAME_CAPTURE/IMG/"
            # Connect to the camera
            camera_1 = self.connect_to_camera(self.CAM1)
            if camera_1:
                camera_1.Open()

                # Only set nodes if they exist; remove GigE-specific nodes for USB cameras
                self.set_camera_node(camera_1, "ExposureTime", 10000)  # Example setting for exposure
                self.set_camera_node(camera_1, "AcquisitionFrameRateEnable", True)
                self.set_camera_node(camera_1, "AcquisitionFrameRate", 30)

                # Start image grabbing
                #camera_1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                # # Adjust camera settings
                # camera_1.GevSCPSPacketSize.SetValue(1500)
                # camera_1.GevSCPD.SetValue(1000)
                # camera_1.GevSCFTD.SetValue(1000)
                # camera_1.AcquisitionFrameRateEnable = True
                # camera_1.AcquisitionFrameRate.SetValue(30)  # Example frame rate

                # Start image grabbing
                camera_1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                try:
                    # Capture and save image
                    self.capture_and_save_image(camera_1, SAVE_IMAGE_DIR_CAM_CAP_HLA, Engine_No,imagcount)
                except Exception as e:
                    print(e)
                    update_Camera_Status_UI("NOT OK")
                    self.logger.debug(f"Exception in capture loop: {str(e)}")
                finally:
                    camera_1.StopGrabbing()  # Stop grabbing images after the operation
                    camera_1.Close()
            else:
                print("Camera is not connected")
                self.logger.debug("Camera_1 is not connected.")
        except Exception as e:
            print(f"Exception during camera setup: {e}")
            update_Camera_Status_UI("NOT OK")
            self.logger.debug(f"Exception in start_capture: {str(e)}")


class PLCCommunication:
    def __init__(self, plcIPAddress, dbReadNumber, dbWriteNumber):
        self.PLC_IP_ADDRESS = plcIPAddress 
        self.DB_READ_NUMBER = dbReadNumber 
        self.DB_WRITE_NUMBER = dbWriteNumber   

    def createConnection(self):
        logger.debug("Inside createCOnnection")
        client = None
        try:
            client = snap7.client.Client()
            client.connect(self.PLC_IP_ADDRESS,0,1)
        except Exception as e:
            update_PLC_Status_UI("NOT OK")
            logger.critical("createCOnnection() Exception is : "+ str(e))
        
        return client

    def isPLCConnected(self, clientConn):
        isConnected = False
        try:
            clientConn.get_connected()
            isConnected = True
        except Exception as e:
            logger.critical("isPLCConnected() Exception is : "+ str(e))
            isConnected = False
        
        return isConnected

    def closePLCConnection(self, clientConn):
        try:
            clientConn.destroy()
        except Exception as e:
            logger.critical("closePLCConnection() Exception is : "+ str(e))

    ''' Read PLC Functions '''
    def readIntFromPLC(self, clientConn, db_col_start_buffer_pos):
        row_data = None
        # db_col_start_buffer_pos = 258
        try:
            if self.isPLCConnected(clientConn) is True:
                db = clientConn.db_read(self.DB_READ_NUMBER, db_col_start_buffer_pos, 2) 
                row_data = util.get_int(db, 0)
                # print(row_data)
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            update_PLC_Status_UI("NOT OK")
            logger.critical("readIntFromPLC() Exception is : "+ str(e))
        return row_data

    def readBoolFromPLC(self, clientConn, db_col_start_buffer_pos):
        row_data = None
        # db_col_start_buffer_pos = 0
        try:
            if self.isPLCConnected(clientConn) is True:
                db = clientConn.db_read(self.DB_READ_NUMBER, db_col_start_buffer_pos, 1) 
                #print(db)
                row_data = util.get_bool(db, 0 ,0)
                # print(f"Read Bool value {row_data}")
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            logger.critical("readBoolFromPLC() Exception is : "+ str(e))
        return row_data

    def readDoubleFromPLC(self, clientConn, db_col_start_buffer_pos):
        row_data = None
        # db_col_start_buffer_pos = 260
        try:
            if self.isPLCConnected(clientConn) is True:
                db = clientConn.db_read(self.DB_READ_NUMBER, db_col_start_buffer_pos, 4) 
                #print(db)
                row_data = util.get_dint(db, 0)
                print(f"Read Double value {row_data}")
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            logger.critical("readDoubleFromPLC() Exception is : "+ str(e))
        return row_data

    def readStringFromPLC(self, clientConn, db_col_start_buffer_pos):
        row_data = None
        # db_col_start_buffer_pos = 2
        try:
            if self.isPLCConnected(clientConn) is True:
                db = clientConn.db_read(self.DB_READ_NUMBER, db_col_start_buffer_pos, 256) 
                #print(db)
                row_data = util.get_string(db, 0)
                # print(f"Read String value {row_data}")

            else:
                logger.debug("PLC not connected")
        except Exception as e:
            logger.critical("readStringFromPLC() Exception is : "+ str(e))
        return row_data

    def readStringFromPLC_1(self, clientConn, db_col_start_buffer_pos):
        row_data = None
        # db_col_start_buffer_pos = 2
        try:
            if self.isPLCConnected(clientConn) is True:
                db = clientConn.db_read(self.DB_READ_NUMBER, db_col_start_buffer_pos, 256) 
                #print(db)
                row_data = util.get_char(db, 0)
                #print(f"Read String value {row_data}")
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            logger.critical("readStringFromPLC() Exception is : "+ str(e))
        return row_data

    ''' Write PLC Functions '''        
    def writeBoolToPLC(self, clientConn, db_col_start_buffer_pos , bool_value):
        try:
            if self.isPLCConnected(clientConn) is True:
                data = bytearray(1)
                util.set_int(data,db_col_start_buffer_pos,bool_value)
                clientConn.db_write(self.DB_WRITE_NUMBER,db_col_start_buffer_pos,data)
                # print("Writing Done")
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            logger.critical("writeBoolToPLC() Exception is : "+ str(e))
            print("writeBoolToPLC() Exception is : "+ str(e))

    def writeIntToPLC(self, clientConn, db_col_start_buffer_pos , int_value):
        try:
            if self.isPLCConnected(clientConn) is True:
                data = bytearray(2)
                util.set_int(data,db_col_start_buffer_pos,int_value)
                clientConn.db_write(self.DB_WRITE_NUMBER,db_col_start_buffer_pos,data)
                if db_col_start_buffer_pos == 0 or db_col_start_buffer_pos == 4:
                    logger.debug(f"Written to PLC for buffer {db_col_start_buffer_pos} and value {data} is success")
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            update_PLC_Status_UI("NOT OK")
            logger.critical("writeIntToPLC() Exception is : "+ str(e))

    def writeDoubleToPLC(self, clientConn, db_col_start_buffer_pos, double_value):
        try:
            if self.isPLCConnected(clientConn) is True:
                data = bytearray(4)
                util.set_dint(data,db_col_start_buffer_pos,double_value)
                clientConn.db_write(self.DB_WRITE_NUMBER,db_col_start_buffer_pos,data)
                # print("Writing Done")
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            logger.critical("writeDoubleToPLC() Exception is : "+ str(e))
            print("writeDoubleToPLC() Exception is : "+ str(e))

    def writeStringToPLC(self, clientConn, db_col_start_buffer_pos,string_value):
        try:
            if self.isPLCConnected(clientConn) is True:
                SPACE_START = "   "
                SPACE_END = "                                                     "
                concat_string = SPACE_START + string_value + SPACE_END
                byte_value = concat_string.encode('utf-8')
                clientConn.db_write(self.DB_WRITE_NUMBER,db_col_start_buffer_pos, data=bytearray(byte_value))
                # str1 = byte_value.decode('UTF-8')  
                # print("Writing Done : "+ str1)
            else:
                logger.debug("PLC not connected")
        except Exception as e:
            logger.critical("writeStringToPLC() Exception is : "+ str(e))

    def handle_position(self,Position,Engine_No,capture_list,imageounter):
        try:    
            #print("postion is ====================",Position)
            lastPostion_is = False
           # time.sleep(1)            
            if Position == True:
                #plcCommunicationObj.writeIntToPLC(clientConn,8,0)
                CAPTURE_FLAG = False                
                # #===================IMAGE SAVE  FOR SEALNT AND POSTION IS 1 ======================#

                camera_capture = CameraCapture(DB_HOST, DB_USER, DB_PASS, DB_NAME, CAM1)
                time.sleep(0.1)
                capture_list.append(Position)
                imagcount = len(capture_list)
                camera_capture.start_capture(Engine_No,CAPTURE_FLAG,imagcount)
                
                imageounter += 1  # Increment the image counter after capturing
                print("len(capture_list)==========================================================",len(capture_list))
                EngineType = plcCommunicationObj.readIntFromPLC(clientConn,514)
                print("EngineType number is=========",EngineType)
                update_Camera_Status_UI("OK")
                logger.debug(f"len(capture_list) is {len(capture_list)} and engine number is {Engine_No}")
                #time.sleep(2)
                if EngineType in [3,5]:

                    if len(capture_list) == 1:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 1 true")
                        time.sleep(0.4)
                        print("=========================== process 01 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)

                    if len(capture_list) == 2:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 2 true")
                        time.sleep(0.4)
                        print("=========================== process 02 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)

                    if len(capture_list) == 3:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 3 true")
                        time.sleep(0.4)
                        print("=========================== process 03 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)                     

                    if len(capture_list) == 4:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 4 true")
                        time.sleep(0.4)
                        print("=========================== process 04 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                        

                    if len(capture_list) == 5:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 5 true")
                        time.sleep(0.4)
                        print("=========================== process 05 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                       

                    if len(capture_list) == 6:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 6 true")
                        time.sleep(0.4)
                        print("=========================== process 06 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                        
                    
                    if len(capture_list) == 7:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 7 true")
                        time.sleep(0.4)
                        print("=========================== process 07 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                        
                    
                    if len(capture_list) == 8:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 8 true")
                        time.sleep(0.4)
                        print("=========================== process 08 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                        
                    
                    if len(capture_list) >=8:
                        while True:
                            print("in while capture list is ",len(capture_list))
                            #Position_CamCap_HLA_Sealnt, IS_PROCESS_CAM_CAP_SEALNT, CAM_CAP_SEALNT_STATUS = CamCap_Sealant_result_status()#getInferenceTrigger
                            CAM_CAP_HLA = "OK"
                            IS_PROCESS_CAM_CAP_HLA = 2
                            if IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="OK":
                                imageounter = 0
                                CAPTURE_FLAG = True

                                plcCommunicationObj.writeIntToPLC(clientConn,0,1)
                                time.sleep(1)
                                plcCommunicationObj.writeIntToPLC(clientConn,0,0)

                                logger.debug(f"Engine Number is :{Engine_No} and CamCap overall ok Status sent to plc")

                                CAM_CAP_HLA = 0
                                IS_PROCESS_CAM_CAP_SEALNT = 0
                                IS_PROCESS_CAM_CAP_HLA = "NO"
                                #Update_ClearImageResult(CAM_CAP_HLA,IS_PROCESS_CAM_CAP_SEALNT,IS_PROCESS_CAM_CAP_HLA)

                                capture_list.clear()
                                capture_list = []
                                print("8 POSSSSS OK SENT TO PLC")
                                break   #OK Register 

                            elif IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="NOT OK":
                                CAPTURE_FLAG = True
                                plcCommunicationObj.writeIntToPLC(clientConn,0,1)
                                time.sleep(1)
                                plcCommunicationObj.writeIntToPLC(clientConn,0,0)

                                logger.debug(f"Engine Number is :{Engine_No} and CamCap overall not ok Status sent to plc")

                                Position_CamCap_HLA_Sealnt = 0
                                IS_PROCESS_CAM_CAP_SEALNT = 0
                                CAM_CAP_SEALNT_STATUS = "NO"
                                Update_ClearImageResult(Position_CamCap_HLA_Sealnt,IS_PROCESS_CAM_CAP_SEALNT,CAM_CAP_SEALNT_STATUS)

                                capture_list.clear()
                                capture_list = []
                                print("8 POSSSSS NOT OK SENT TO PLC")
                                break
                            
                            else:
                                print("Else Position [1] IS_PROCESS_CAM_CAP_SEALNT ")
                                time.sleep(0.2)
                                continue

                else:
                    if len(capture_list) == 1:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 1 true")
                        time.sleep(0.4)
                        print("=========================== process 01 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)

                    if len(capture_list) == 2:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 2 true")
                        time.sleep(0.4)
                        print("=========================== process 02 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)

                    if len(capture_list) == 3:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 3 true")
                        time.sleep(0.4)
                        print("=========================== process 03 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)

                    if len(capture_list) == 4:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 4 true")
                        time.sleep(0.4)
                        print("=========================== process 04 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)

                    if len(capture_list) == 5:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 5 true")
                        time.sleep(0.4)
                        print("=========================== process 05 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)

                    if len(capture_list) == 6:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"len(capture_list) is {len(capture_list)} condtion in 6 true")
                        time.sleep(0.4)
                        print("=========================== process 06 =============================")
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)


                      
                    if len(capture_list) >=6:
                        while True:
                            print("in while capture list is ",len(capture_list))
                            #Position_CamCap_HLA_Sealnt, IS_PROCESS_CAM_CAP_SEALNT, CAM_CAP_SEALNT_STATUS = CamCap_Sealant_result_status()#getInferenceTrigger
                            CAM_CAP_HLA = "OK"
                            IS_PROCESS_CAM_CAP_HLA = 2#"OK"
                            if IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="OK":
                                imageounter = 0
                                CAPTURE_FLAG = True
                                plcCommunicationObj.writeIntToPLC(clientConn,0,1)
                                time.sleep(1)
                                plcCommunicationObj.writeIntToPLC(clientConn,0,0)
                                logger.debug(f"Engine Number is :{Engine_No} and CamCap ok Status sent to plc")

                                CAM_CAP_HLA = "OK"
                                IS_PROCESS_CAM_CAP_HLA = 2

                                capture_list.clear()
                                capture_list = []
                                print("6 POSSSSS OK SENT TO PLC")
                                break   #OK Register 

                            elif IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="NOT OK":
                                CAPTURE_FLAG = True
                                plcCommunicationObj.writeIntToPLC(clientConn,0,1)
                                time.sleep(1)
                                plcCommunicationObj.writeIntToPLC(clientConn,0,0)

                                logger.debug(f"Engine Number is :{Engine_No} and CamCap not ok Status sent to plc")

                                CAM_CAP_HLA = "NOT OK"
                                IS_PROCESS_CAM_CAP_HLA = 2
                            # Update_ClearImageResult(Position_CamCap_HLA_Sealnt,IS_PROCESS_CAM_CAP_SEALNT,CAM_CAP_SEALNT_STATUS)

                                capture_list.clear()
                                capture_list = []
                                print("6 POSSSSS NOT OK SENT TO PLC")
                                break
                            
                            else:
                                print("Else Position [1] IS_PROCESS_CAM_CAP ")
                                time.sleep(0.2)
                                continue
                
                    

            else:
                print("postion is ====:",Position)
           
                
        except Exception as e:
            print("HANDALE POSTION IS :",e)
            print(traceback.format_exc())

def CamCap_Sealant_result_status():
    Position, IS_PROCESS, Status = "", "", ""
    db_fetch = None
    cur = None
    try:
        db_fetch = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_fetch.cursor()
        query = "select * from CamCap_Sealant_result_status;"
        cur.execute(query)
        data_set = cur.fetchall()
        for row in range(len(data_set)):
            Position = int(data_set[row][2])
            IS_PROCESS = int(data_set[row][3])
            Status = str(data_set[row][6])
        cur.close()
    except Exception as e:
        print("CamCap_Sealant_result_status is :",e)
      
    finally:
        if cur is not None:
            cur.close()
        if db_fetch is not None:
            db_fetch.close() 
    return Position,IS_PROCESS,Status

def Update_ClearImageResult(Position_CamCap_HLA_Sealnt,IS_PROCESS_CAM_CAP_SEALNT,CAM_CAP_SEALNT_STATUS):
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        cur = db_update.cursor()
        query = "UPDATE configuration_table SET Postion = %s, Is_process = %s, Status = %s WHERE ID = 1"
        cur.execute(query, (Position_CamCap_HLA_Sealnt,IS_PROCESS_CAM_CAP_SEALNT,CAM_CAP_SEALNT_STATUS))
        db_update.commit()
    except Exception as e:
        print("update_Engine_No() Exception is : "+ str(e))

    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close()  


def getInferenceTrigger():
    Engine_no, Position = "", "", None,"","",""
    db_fetch = None
    cur = None
    try:
        db_fetch = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_fetch.cursor()
        query = "select * from engine_number_table;"
        cur.execute(query)
        data_set = cur.fetchall()
        for row in range(len(data_set)):
            Engine_no = str(data_set[row][1])
            Position = int(data_set[row][2])
        cur.close()
    except Exception as e:
        print("getInferenceTrigger",e)
        # logger.error(f"Error in getting inference trigger: {e}")
        # return "", "", None
    finally:
        if cur is not None:
            cur.close()
        if db_fetch is not None:
            db_fetch.close() 
    return Engine_no, Position

def update_Engine_No_Cam2(Engine_No,Position):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        query = "UPDATE config_table SET Engine_no = %s, Position = %s WHERE ID = 1"
        cur.execute(query, (Engine_No,Position))
        db_update.commit()
    except Exception as e:
        print("update_Engupdate_Engine_No_Cam2ine_No() Exception is : "+ str(e))
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close()        


def update_Engine_No(Engine_No,Position):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        query = "UPDATE config_table SET Engine_no = %s, Position = %s WHERE ID = 1"
        cur.execute(query, (Engine_No,Position))
        db_update.commit()
    except Exception as e:
        print("update_Engine_No () Exception is : "+ str(e))
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close() 

def update_Engine_Name(Engine_No,engine_type):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        query = "UPDATE engine_number_table SET ENGINE_NUMBER = %s, ENGINE_TYPE = %s WHERE ID = 1 "
        cur.execute(query, (Engine_No,engine_type))
       # cur.execute(query)
        db_update.commit()
    except Exception as e:
        print("update_Engine_Name() Exception is : "+ str(e))
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close() 


#========================== PLCCommunication ==============================#
if __name__ == "__main__":
    plcIPAddress = '192.168.3.201'
    debReadNumber = 104
    dbWriteNumber = 108

    plcCommunicationObj = PLCCommunication(plcIPAddress, debReadNumber, dbWriteNumber) 
    clientConn = plcCommunicationObj.createConnection()

    Engine_No=""
    result_list_ok_nok = []
    capture_list = []
    write_list= []
    imageounter = 0
    plcCommunicationObj.writeIntToPLC(clientConn,0,0)
    engine = "3_Cylinder"
    CaptureBit = False
    update_PLC_Status_UI("NOT OK")   #UPDATE_FRAME_CAPTURE_PROCESS
    update_Camera_Status_UI("NOT OK")
    homeposition = ''
    while True:
        totalstarttime = time.time()
        today_date_folder = datetime.datetime.now().strftime("%Y-%m-%d")
        SAVE_IMAGE_DIR_CAM_CAP_HLA = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/FRAME_CAPTURE/IMG/"

        positiontime1=time.time()
        Position = plcCommunicationObj.readBoolFromPLC(clientConn,512)
        EngineType = plcCommunicationObj.readIntFromPLC(clientConn,514)
        Engine_No = plcCommunicationObj.readStringFromPLC(clientConn, engineNumberStringBufferPosition)
        homeposition = plcCommunicationObj.readBoolFromPLC(clientConn,782)
        #print("CaptureBit is=============================:",CaptureBit)
        update_PLC_Status_UI("OK")
        if Position is True and CaptureBit == False:
            #plcCommunicationObj.writeIntToPLC(clientConn,0,0)
            print("postion is =========",Position)
            logger.debug(f"Position is {Position} and CaptureBit is {CaptureBit} for engine number is {Engine_No}")
            try:
                Engine_No = plcCommunicationObj.readStringFromPLC(clientConn, engineNumberStringBufferPosition)
                #print("Engine_No IS============= :",Engine_No)
                CaptureBit = True
                #update_PLC_postion_Status("1")
                #time.sleep(0.3)
            except Exception as e:
                print(f"Error occurred: {e}")
                logger.debug(f"except Exception for engine number is {Engine_No}")
                pass
        else:
            CaptureBit = False
            print("else Position is=============================:",Position)
            update_PLC_postion_Status("0")
            continue    
        ENGINENAME=Engine_No.replace(' ','_')
        ### 5model 4 = 3cy-6pos, 1 = 3cy-6pos, 5 = 4cy-8pos,  3 = 4cy-8pos,
        EngineType = plcCommunicationObj.readIntFromPLC(clientConn,514)
        if EngineType in [3,5]:
            engine = "4_Cylinder"
            if len(capture_list) ==8:
                imageounter = 0
                capture_list = []
                result_list_ok_nok= []
                write_list= []
                capture_list.clear()
                result_list_ok_nok.clear()
                write_list.clear()      
        
        if EngineType in [4,1]:
            engine = "3_Cylinder"
            if len(capture_list) ==6:
                imageounter = 0
                capture_list = []
                result_list_ok_nok= []
                write_list= []
                capture_list.clear()
                result_list_ok_nok.clear()
                write_list.clear()

        if homeposition is True:
            print(f"homeposition value is =============={homeposition}")   
            capture_list.clear()
            capture_list =[]
               

        if ENGINENAME is None or ENGINENAME == '':
            print("ENGINENAME is ================",ENGINENAME)
            continue


        update_Engine_Name(ENGINENAME,engine)
        update_PLC_Status_UI("OK")
        plcCommunicationObj.handle_position(Position,ENGINENAME,capture_list,imageounter)
       
        totalend=time.time()
        #print("toatltime",totalend-totalstarttime)     cycle deee  #update_PLC_postion_Status   FRAMECAPTURE_STATUS
