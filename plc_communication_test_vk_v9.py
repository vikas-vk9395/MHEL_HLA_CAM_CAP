#!/usr/bin/python3
from tendo import singleton
single_instance = singleton.SingleInstance()
import os
import PySpin
import time 
import cv2
import numpy as np
import pymysql
import logging
import traceback
from logging.handlers import RotatingFileHandler
import random
import multiprocessing
from multiprocessing.dummy import Process
import subprocess
from subprocess import Popen, PIPE
import threading


import snap7
from snap7 import util


import traceback
from logging.handlers import RotatingFileHandler
import datetime


''' Initializing Logger '''
fileName = os.path.basename(__file__).split(".")[0]
logger = None
''' Initializing Logger '''
# Create the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from logging.handlers import TimedRotatingFileHandler
# Define the log file format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Create a TimedRotatingFileHandler for daily rotation
log_file = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/FRAME_CAPTURE/"+os.path.basename(__file__[:-2])+"log"
file_handler = TimedRotatingFileHandler(log_file, when='midnight', backupCount=10)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.debug("Engine Inference module initialized")
DEBUG_MODE = False

# Set up logger
logger = logging.getLogger("FLAIR_SEQ_CAM_CAP_HLA")
logger.setLevel(logging.DEBUG)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_fh = logging.FileHandler("FLAIR_SEQ_CAM_CAP_HLA.log", mode='a')
logger_fh.setFormatter(log_format)
logger.addHandler(logger_fh)




DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "insightzz@123"
DB_NAME = "hla_cam_cap_db"

CAM1 ='24117731'
processID = os.getpid()



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


def CamCap_Sealant_result_status():
    IS_PROCESS, OverallOk, CamCapstatus, HLA_status = "", "", "", ""
    db_fetch = None
    cur = None
    try:
        db_fetch = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_fetch.cursor()
        query = "select * from engine_status_config;"
        cur.execute(query)
        data_set = cur.fetchall()
        for row in range(len(data_set)):
            OverallOk = str(data_set[row][2])
            CamCapstatus = str(data_set[row][3])
            HLA_status = str(data_set[row][4])
            IS_PROCESS = int(data_set[row][5])
        cur.close()
    except Exception as e:
        print("engine_status_config is :",e)
      
    finally:
        if cur is not None:
            cur.close()
        if db_fetch is not None:
            db_fetch.close() 
    return IS_PROCESS, OverallOk, CamCapstatus, HLA_status

def Update_ClearImageResult(IS_PROCESS, OverallOk, CamCapstatus, HLA_status):
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        cur = db_update.cursor()
        query = "UPDATE engine_status_config SET IS_PROCESS_INF = %s, OVREALL_STATUS = %s, OVERALL_CAM_CAP_RESULT = %s, OVERALL_HLA_RESULT = %s WHERE ID = 1"
        cur.execute(query, (IS_PROCESS, OverallOk, CamCapstatus, HLA_status))
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




def configure_exposure(camera, exposure_value):
    print('*** CONFIGURING EXPOSURE ***\n')

    try:
        result = True
        if camera.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to disable automatic exposure. Aborting...')
            return False

        camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        print('Automatic exposure disabled...')

        if camera.ExposureTime.GetAccessMode() != PySpin.RW:
            print('Unable to set exposure time. Aborting...')
            return False

        # Ensure desired exposure time does not exceed the maximum
        exposure_time_to_set = exposure_value
        exposure_time_to_set = min(camera.ExposureTime.GetMax(), exposure_time_to_set)
        camera.ExposureTime.SetValue(exposure_time_to_set)
        print('Shutter time set to %s us...\n' % exposure_time_to_set)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

class PLCCommunication:
        
    #========================= LOG ===========================#
    import logging
    from logging import handlers
    logger = logging.getLogger("MAHINDRA_CAM_CAP_HLA_FRAME_CAPTURE_v1")
    logger.setLevel(logging.DEBUG)

    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fileName = os.path.basename(__file__).split(".")[0]+".log"
    log_fl=logging. handlers.RotatingFileHandler(fileName,maxBytes=1048576,backupCount=5) # 1MB log files max
    logger_Fh=logging.FileHandler("MAHINDRA_CAM_CAP_HLA_FRAME_CAPTURE_v1.log", mode='a') # 1MB log files max

    logger_Fh.setFormatter(log_format)
    logger.addHandler(logger_Fh)
    logger.debug("Top Module Initialized")
    
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
            self.createConnection()
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
            self.createConnection()
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



class ImageCapture():
    Frame=None
    stopThread=False

    def __init__(self):
        self.data_collection=False
        try:
            t1=threading.Thread(target=self.initCam)
            t1.start()
        except Exception as e:
            logger.error(traceback.format_exc())
            
    def configure_exposure(self,cam, exposure_value):
        try:
            result = True
            if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                return False

            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

            if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                return False

            exposure_time_to_set = exposure_value
            exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
            cam.ExposureTime.SetValue(exposure_time_to_set)
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            logger.error(traceback.format_exc())
            logger('Error: %s' % ex)
            result = False
        return result
    
    def acquire_images(self,cam_list):
        try:
            result = True
            for i, cam in enumerate(cam_list):
                node_device_serial_number = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
                device_serial_number = node_device_serial_number.GetValue()

                tlStreamSetup=cam.GetTLStreamNodeMap()
                resendFramesNode=PySpin.CBooleanPtr(tlStreamSetup.GetNode("StreamPacketResendEnable"))
                if PySpin.IsAvailable(resendFramesNode) and PySpin.IsReadable(resendFramesNode) and PySpin.IsWritable(resendFramesNode):
                    resendFramesNode.SetValue(False)

                deviceThroughput=PySpin.CIntegerPtr(cam.GetNodeMap().GetNode('DeviceLinkThroughputLimit'))

                if device_serial_number in [CAM1]:
                    if PySpin.IsAvailable(deviceThroughput) and PySpin.IsReadable(deviceThroughput):
                        device_throughput = 156160#15616000 
                        deviceThroughput.SetValue(device_throughput)
                        
                node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
                if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                    logger.error(traceback.format_exc())
                    return False

                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                        node_acquisition_mode_continuous):
                    logger.error(traceback.format_exc())
                    return False

                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                cam.BeginAcquisition()

            image_incomplete_counter = 0
            counter = 0
            capturedList=[]

            Engine_No=""
            result_list_ok_nok = []
            capture_list = []
            write_list= []
            imageounter = 0
            #plcCommunicationObj.writeIntToPLC(clientConn,0,0)
            engine = "3_Cylinder"
            CaptureBit = False
            update_PLC_Status_UI("OK")
            update_Camera_Status_UI("OK")
            homeposition = ''

            while not self.stopThread:
                try:
                    totalstarttime = time.time()
                    today_date_folder = datetime.datetime.now().strftime("%Y-%m-%d")
                    SAVE_IMAGE_DIR_CAM_CAP_HLA = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/FRAME_CAPTURE/IMG/"

                    positiontime1=time.time() # try:
                    time.sleep(0.1)
                    Position = plcCommunicationObj.readIntFromPLC(clientConn,784)  #784 #784
                    Engine_No = plcCommunicationObj.readStringFromPLC(clientConn, engineNumberStringBufferPosition)
                    homeposition = plcCommunicationObj.readBoolFromPLC(clientConn,782)
                    actNotReached = plcCommunicationObj.readBoolFromPLC(clientConn,786)
                    #print("CaptureBit is=============================:",CaptureBit)
                    #update_PLC_Status_UI("OK")
                    #plcCommunicationObj.writeIntToPLC(clientConn,0,1)
                    print(f"homeposition value is =============={homeposition}")  
                    if homeposition is True:
                        
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0) 
                        capture_list.clear()
                        capture_list =[]

                    if actNotReached is True:
                        plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                        time.sleep(0.4)
                        #print(f"actNotReached value is =============={actNotReached}")   
                        plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                        logger.debug(f"================engine number is {Engine_No} actNotReached is True:")
                    
                    if Position != 0:
                        #plcCommunicationObj.writeIntToPLC(clientConn,0,0)
                        print("postion is =========",Position)
                        logger.debug(f"Position is {Position} and for engine number is {Engine_No}")
                        try:
                            Engine_No = plcCommunicationObj.readStringFromPLC(clientConn, engineNumberStringBufferPosition)
                            #print("Engine_No IS============= :",Engine_No)
                            CaptureBit = True
                            start_time = time.time()
                            
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            logger.debug(f"except Exception for engine number is {Engine_No}")
                            pass
                    else:
                        CaptureBit = False
                       # print("else Position is=============================:",Position)
                    # update_PLC_postion_Status("0")
                        continue 

                    ENGINENAME=Engine_No.replace(' ','_')
                    ### 5model 4 = 3cy-6pos, 1 = 3cy-6pos, 5 = 4cy-8pos,  3 = 4cy-8pos

                    EngineType = plcCommunicationObj.readIntFromPLC(clientConn,514)
                    if EngineType in [3,5]:
                        engine = "4_Cylinder"
                    
                    if EngineType in [4,1]:
                        engine = "3_Cylinder"


                    if homeposition is True:
                        print(f"homeposition value is =============={homeposition}")   
                        capture_list.clear()
                        capture_list =[]
                        

                    if ENGINENAME is None or ENGINENAME == '':
                        print("ENGINENAME is ================",ENGINENAME)
                        continue


                    update_Engine_Name(ENGINENAME,engine)
                    
                    print("cyclet postion is ==================================",Position)

                    # Constants
                    IMGATHRANE_FILE = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/ALGORITHAM/IMG/"
                    SAVE_IMAGE_DIR_CAM_CAP_HLA = "C:/Users/mahindra/insightzz/ALGORITHM_NEW_HLA/FRAME_CAPTURE/IMG/"
                    PLC_OK = 1
                    PLC_NOT_OK = 2
                    PLC_REGISTER_ENGINE = 6
                    update_PLC_Status_UI("OK")
                    #time.sleep(1)
                    if engine == "4_Cylinder":
                        if Position in [1,2,3,4,5,6,7,8] and Position not in capture_list:
                            time.sleep(0.1)
                            for i, cam in enumerate(cam_list):
                                # Ensure camera is not streaming
                                if cam.IsStreaming():
                                    cam.EndAcquisition()
                                # Initialize camera if not already initialized
                                if not cam.IsInitialized():
                                    cam.Init()

                                # Get the node for acquisition mode
                                node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))

                                # Check if the node is available and writable
                                if not PySpin.IsAvailable(node_acquisition_mode):
                                    print(f"AcquisitionMode node is not available for camera {i}")
                                    logger.error(f"AcquisitionMode node is not available for camera {i}")
                                    return False
                                
                                if not PySpin.IsWritable(node_acquisition_mode):
                                    print(f"AcquisitionMode node is not writable for camera {i}")
                                    logger.error(f"AcquisitionMode node is not writable for camera {i}")
                                    return False

                                # If available, proceed to set the mode to 'Continuous'
                                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                                if not PySpin.IsAvailable(node_acquisition_mode_continuous):
                                    print(f"Continuous mode is not available for camera {i}")
                                    logger.error(f"Continuous mode is not available for camera {i}")
                                    return False

                                if not PySpin.IsReadable(node_acquisition_mode_continuous):
                                    print(f"Continuous mode is not readable for camera {i}")
                                    logger.error(f"Continuous mode is not readable for camera {i}")
                                    return False

                                # Set acquisition mode to Continuous
                                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                                print(f"Camera {i}: Acquisition mode set to Continuous")
                                logger.info(f"Camera {i}: Acquisition mode set to Continuous")


                                cam.BeginAcquisition()  # Start continuous acquisition
                                image_result = cam.GetNextImage(100)

                                if image_result.IsIncomplete():
                                    image_incomplete_counter += 1
                                    print(f"Device {device_serial_number}: Image incomplete with status {image_result.GetImageStatus()}")
                                    logger.error(f"Device {device_serial_number}: Image incomplete with status {image_result.GetImageStatus()}")
                                    logger.error(traceback.format_exc())
                                else:
                                   # plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                                    # Reset incomplete counter and process the image
                                    image_incomplete_counter = 0
                                    image_data = image_result.GetNDArray()

                                    # Convert the image to BGR8 format
                                    processor = PySpin.ImageProcessor()
                                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                                    image_converted_np = image_converted.GetNDArray()

                                    # Save the processed image
                                    TodaysDate = datetime.datetime.now().strftime('%Y_%m_%d')
                                    image_folder = os.path.join(SAVE_IMAGE_DIR_CAM_CAP_HLA, TodaysDate, Engine_No)
                                    os.makedirs(image_folder, exist_ok=True)

                                    # Unique filename with position and timestamp
                                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                    image_filename = f"POS_{Position}.jpg"
                                    image_path = os.path.join(IMGATHRANE_FILE, image_filename)
                                    
                                    # Rotate the image if needed and save
                                    img_rotated = cv2.rotate(image_converted_np, cv2.ROTATE_180)
                                    cv2.imwrite(image_path, img_rotated)
                                    image_path_new = os.path.join(image_folder, image_filename)
                                    cv2.imwrite(image_path_new, img_rotated)
                                    print(f"Image captured and saved: {image_path}")
                                    plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                                    # Add position to capture list and release the image buffer
                                    capture_list.append(Position)

                                    logger.debug(f"Captured image for Position {Position}, saved at {image_path}")
                                    # Release the current image to free memory
                                    image_result.Release()

                                    # Add a small delay to allow for new frame acquisition (optional)
                                    time.sleep(0.1)

                                    plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                                   # time.sleep(0.3)
                                    if len(capture_list) >=8:
                                        while True:
                                            print("in while capture list is ",len(capture_list))
                                            #IS_PROCESS, OverallOk, CamCapstatus, HLA_status = CamCap_Sealant_result_status()#getInferenceTrigger
                                            CAM_CAP_HLA = "OK"
                                            IS_PROCESS_CAM_CAP_HLA = 2
                                            if IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="OK":
                                                imageounter = 0
                                                CAPTURE_FLAG = True
                                                plcCommunicationObj.writeIntToPLC(clientConn,0,1)#overallok
                                                plcCommunicationObj.writeIntToPLC(clientConn,10,1)#overallokcamcap
                                                plcCommunicationObj.writeIntToPLC(clientConn,8,1)#overallokcamcaphla
                                                logger.debug(f"4cylinder Engine Number is :{Engine_No} and CamCap overall ok Status sent to plc")

                                                CAM_CAP_HLA = 0
                                                IS_PROCESS_CAM_CAP_SEALNT = 0
                                                IS_PROCESS_CAM_CAP_HLA = "NO"
                                                #Update_ClearImageResult(IS_PROCESS, OverallOk, CamCapstatus, HLA_status)

                                                capture_list.clear()
                                                capture_list = []
                                                print("8 POSSSSS OK SENT TO PLC")
                                            #  time.sleep(5)
                                                break   #OK Register 

                                            elif IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="NOT OK":
                                                CAPTURE_FLAG = True
                                                plcCommunicationObj.writeIntToPLC(clientConn,0,1)#overallok
                                                plcCommunicationObj.writeIntToPLC(clientConn,10,1)#overallokcamcap
                                                plcCommunicationObj.writeIntToPLC(clientConn,8,1)#overallokcamcaphla
                                            
                                                # plcCommunicationObj.writeIntToPLC(clientConn,0,2)#overallok
                                                # plcCommunicationObj.writeIntToPLC(clientConn,10,2)#overallokcamcap
                                                # plcCommunicationObj.writeIntToPLC(clientConn,8,2)#overallokcamcaphla

                                                logger.debug(f"4cylinder Engine Number is :{Engine_No} and CamCap overall not ok Status sent to plc")

                                                Position_CamCap_HLA_Sealnt = 0
                                                IS_PROCESS_CAM_CAP_SEALNT = 0
                                                CAM_CAP_SEALNT_STATUS = "NO"
                                            # Update_ClearImageResult(IS_PROCESS, OverallOk, CamCapstatus, HLA_status)

                                                capture_list.clear()
                                                capture_list = []
                                                print("8 POSSSSS NOT OK SENT TO PLC")
                                            #  time.sleep(5)
                                                break
                                        
                                            else:
                                                print("Else Position [1] IS_PROCESS_CAM_CAP_SEALNT ")
                                                time.sleep(0.2)
                                                #continue
                 
                    if engine == "3_Cylinder":                 
                        if Position in [1,2,3,4,5,6] and Position not in capture_list:                         
                            for i, cam in enumerate(cam_list):
                                # Ensure camera is not streaming
                                time.sleep(0.1)
                                if cam.IsStreaming():
                                    cam.EndAcquisition()

                                # Initialize camera if not already initialized
                                if not cam.IsInitialized():
                                    cam.Init()

                                # Get the node for acquisition mode
                                node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))

                                # Check if the node is available and writable
                                if not PySpin.IsAvailable(node_acquisition_mode):
                                    print(f"AcquisitionMode node is not available for camera {i}")
                                    logger.error(f"AcquisitionMode node is not available for camera {i}")
                                    return False
                                
                                if not PySpin.IsWritable(node_acquisition_mode):
                                    print(f"AcquisitionMode node is not writable for camera {i}")
                                    logger.error(f"AcquisitionMode node is not writable for camera {i}")
                                    return False

                                # If available, proceed to set the mode to 'Continuous'
                                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                                if not PySpin.IsAvailable(node_acquisition_mode_continuous):
                                    print(f"Continuous mode is not available for camera {i}")
                                    logger.error(f"Continuous mode is not available for camera {i}")
                                    return False

                                if not PySpin.IsReadable(node_acquisition_mode_continuous):
                                    print(f"Continuous mode is not readable for camera {i}")
                                    logger.error(f"Continuous mode is not readable for camera {i}")
                                    return False

                                # Set acquisition mode to Continuous
                                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                                print(f"Camera {i}: Acquisition mode set to Continuous")
                                logger.info(f"Camera {i}: Acquisition mode set to Continuous")


                                cam.BeginAcquisition()  # Start continuous acquisition
                                image_result = cam.GetNextImage(100)
                                
                                if image_result.IsIncomplete():
                                    image_incomplete_counter += 1
                                    print(f"Device {device_serial_number}: Image incomplete with status {image_result.GetImageStatus()}")
                                    logger.error(f"Device {device_serial_number}: Image incomplete with status {image_result.GetImageStatus()}")
                                    logger.error(traceback.format_exc())
                                else:
                                    
                                    plcCommunicationObj.writeIntToPLC(clientConn,6,1)
                                    # Reset incomplete counter and process the image
                                    image_incomplete_counter = 0
                                    image_data = image_result.GetNDArray()
                                    
                                    # Convert the image to BGR8 format
                                    processor = PySpin.ImageProcessor()
                                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                                    image_converted_np = image_converted.GetNDArray()

                                    # Save the processed image
                                    TodaysDate = datetime.datetime.now().strftime('%Y_%m_%d')
                                    image_folder = os.path.join(SAVE_IMAGE_DIR_CAM_CAP_HLA, TodaysDate, Engine_No)
                                    os.makedirs(image_folder, exist_ok=True)

                                    # Unique filename with position and timestamp
                                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                    image_filename = f"POS_{Position}.jpg"
                                    image_path = os.path.join(IMGATHRANE_FILE, image_filename)
                                    
                                    # Rotate the image if needed and save
                                    img_rotated = cv2.rotate(image_converted_np, cv2.ROTATE_180)
                                    cv2.imwrite(image_path, img_rotated)
                                    image_path_new = os.path.join(image_folder, image_filename)
                                    cv2.imwrite(image_path_new, img_rotated)
                                    print(f"Image captured and saved: {image_path}")
                                    
                                    # Add position to capture list and release the image buffer
                                    capture_list.append(Position)
                                    logger.debug(f"Captured image for Position {Position}, saved at {image_path}")
                                       # Release the current image to free memory
                                    image_result.Release()
                                    time.sleep(0.1)
                                    # Add a small delay to allow for new frame acquisition (optional)                                    
                                    plcCommunicationObj.writeIntToPLC(clientConn,6,0)
                                    
                                       # image_result.Release() 
                                    if len(capture_list) >=6:
                                            while True:
                                                IS_PROCESS, OverallOk, CamCapstatus, HLA_status = CamCap_Sealant_result_status()
                                                print("in while capture list is ",len(capture_list))
                                                CAM_CAP_HLA = "OK"
                                                IS_PROCESS_CAM_CAP_HLA = 2#"OK"
                                                if IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="OK":
                                                    imageounter = 0
                                                    CAPTURE_FLAG = True
                                                    plcCommunicationObj.writeIntToPLC(clientConn,0,1)#overallok
                                                    plcCommunicationObj.writeIntToPLC(clientConn,10,1)#overallokcamcap
                                                    plcCommunicationObj.writeIntToPLC(clientConn,8,1)#overallokcamcaphla
                                                
                                                    logger.debug(f"3cylinder Engine Number is :{Engine_No} and CamCap ok Status sent to plc")

                                                    CAM_CAP_HLA = "OK"
                                                    IS_PROCESS_CAM_CAP_HLA = 2
                                                    Update_ClearImageResult(IS_PROCESS, OverallOk, CamCapstatus, HLA_status)
                                                    capture_list.clear()
                                                    capture_list = []
                                                    print("6 POSSSSS OK SENT TO PLC")
                                                #  time.sleep(5)
                                                    break   #OK Register 

                                                elif IS_PROCESS_CAM_CAP_HLA == 2 and CAM_CAP_HLA =="NOT OK":
                                                    CAPTURE_FLAG = True
                                                    plcCommunicationObj.writeIntToPLC(clientConn,0,1)#overallok
                                                    plcCommunicationObj.writeIntToPLC(clientConn,10,1)#overallokcamcap
                                                    plcCommunicationObj.writeIntToPLC(clientConn,8,1)#overallokcamcaphla

                                                    #  plcCommunicationObj.writeIntToPLC(clientConn,0,2)#overallok
                                                    # plcCommunicationObj.writeIntToPLC(clientConn,10,2)#overallokcamcap
                                                    # plcCommunicationObj.writeIntToPLC(clientConn,8,2)#overallokcamcaphla
                                                
                                                    logger.debug(f"3cylinder Engine Number is :{Engine_No} and CamCap not ok Status sent to plc")

                                                    CAM_CAP_HLA = "NOT OK"
                                                    IS_PROCESS_CAM_CAP_HLA = 2
                                                    Update_ClearImageResult(IS_PROCESS, OverallOk, CamCapstatus, HLA_status)
                                                    capture_list.clear()
                                                    capture_list = []
                                                    print("6 POSSSSS NOT OK SENT TO PLC")
                                                # time.sleep(5)
                                                    break
                                                
                                                else:
                                                    print("Else Position [1] IS_PROCESS_CAM_CAP ")
                                                    time.sleep(0.2)
                                                    continue
                                                     
                except Exception as e:
                    logger.debug(f"Exception in  3cylinder {len(capture_list)} and engine number is {Engine_No}")
                    print("Exception in 3t while loop :",e)
            

            
        except PySpin.SpinnakerException as ex:
            print(device_serial_number)
            print(traceback.format_exc)
            print(f"Exception in {ex}")
            logger.error("SIDE1 CAM is not connected")

            logger.error('Error: %s' % ex)
            result = False

        return result

    def print_device_info(self,nodemap, cam_num):
        try:
            result = True
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(),
                                    node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

            else:
                logger.error(traceback.format_exc())

        except PySpin.SpinnakerException as ex:
            logger.error(traceback.format_exc())
            return False

        return result

    def getSerialNumber(self,cam):
        device_serial_number = ''
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
        return device_serial_number   

    def run_multiple_cameras(self,cam_list):
        try:
            result = True

            device_list = []
            for i, cam in enumerate(cam_list):
                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                device_list.append(self.getSerialNumber(cam))
                result &= self.print_device_info(nodemap_tldevice, i)
                
            for device in device_list:
                if CAM1 in device_list:
                    print("SIDE1 CAM is connected")
           
            for i, cam in enumerate(cam_list):
                cam.Init()

            result &= self.acquire_images(cam_list)

            for cam in cam_list:
                cam.DeInit()

            del cam

        except PySpin.SpinnakerException as ex:
            logger.error(traceback.format_exc())
            print('Error: %s' % ex)
            result = False

        return result

    def initCam(self):
        result = True

        system = PySpin.System.GetInstance()
        version = system.GetLibraryVersion()

        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()

        if num_cameras == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            logger.error(traceback.format_exc())
            return False

        result = self.run_multiple_cameras(cam_list)

        print('Example complete... \n')

        cam_list.Clear()
        system.ReleaseInstance()

        return result

#========================== PLCCommunication ==============================#
if __name__ == "__main__":

    plcIPAddress = '192.168.3.201'
    debReadNumber = 104
    dbWriteNumber = 108
    plcCommunicationObj = PLCCommunication(plcIPAddress, debReadNumber, dbWriteNumber) 
    clientConn = plcCommunicationObj.createConnection()

    update_PLC_Status_UI("NOT OK")
    update_Camera_Status_UI("NOT OK")
    # Call create_directories before initializing the ImageCapture object
    ImageCapture()

    
