import cv2
import os
import traceback
import pymysql
import datetime
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
import numpy as np
import logging
import ast
import xml.etree.ElementTree as ET
import json
import subprocess
from logging.handlers import TimedRotatingFileHandler

from detectron2.utils.logger import setup_logger
# from future.backports.test.pystone import FALSE
setup_logger()

from PIL import Image
from skimage.io import imread
from PIL import ImageFont, ImageDraw, Image
from fpdf import FPDF
from shapely.geometry import Point, Polygon
from subprocess import Popen, PIPE
from fpdf import FPDF
import os
from PIL import ImageFont, ImageDraw, Image
import datetime
import pymysql
import logging
from logging import handlers
import os

gMailObj = None

os.environ['CUDA_VISIBLE_DEVICES']='0'

DEMO_RUN_FLAG = False

# Load configuration from XML
config_path = "D:/INSIGHTZZ/PRODUCATION_CODE/ALGORITHAM/APP_CONFIG.xml"
if DEMO_RUN_FLAG:
    config_path = "/home/viks/VIKS/PROJECT_ALGORITHAM/MAHINDRA_CHKHAN_GATE-12/HLA_CAM_CAP/SEP_26/ALGORITHAM/APP_CONFIG.xml"

config_parse = ET.parse(config_path)
config_root = config_parse.getroot()
CODE_PATH = config_root[0].text
DB_USER = config_root[1].text
DB_PASS = config_root[2].text
DB_HOST = config_root[3].text
DB_NAME = config_root[4].text
NUMCLASSES = int(config_root[5].text)
DETECTTHRESH = float(config_root[6].text)
SAVED_FOLDER_PATH = config_root[7].text
MASK_MODEL_PATH = config_root[8].text
CONFIG_YAML_FL_PATH = config_root[9].text
json_path = config_root[10].text
DOWNLOAD_PATH = config_root[11].text
NOT_OK_SAVED_FOLDER_PATH = config_root[12].text
NOT_OK_SAVED_FOLDER_PATH = "D:/INSIGHTZZ/PRODUCATION_CODE/ALGORITHAM/NOT_OK_SAVED_FOLDER_PATH/"
NOT_OK_CAP_SAVED_FOLDER_PATH = "D:/INSIGHTZZ/PRODUCATION_CODE/ALGORITHAM/CAP_DEFECT_DATA/"
base_dir = "D:/INSIGHTZZ/PRODUCATION_CODE/ALGORITHAM/LOG/"

if DEMO_RUN_FLAG:
    NOT_OK_SAVED_FOLDER_PATH = "/home/viks/VIKS/PROJECT_ALGORITHAM/MAHINDRA_CHKHAN_GATE-12/HLA_CAM_CAP/SEP_26/ALGORITHAM/DEFECT_DATA/"
    NOT_OK_CAP_SAVED_FOLDER_PATH = "/home/viks/VIKS/PROJECT_ALGORITHAM/MAHINDRA_CHKHAN_GATE-12/HLA_CAM_CAP/SEP_26/ALGORITHAM/CAP_DEFECT_DATA/"


def __loadLablMap__():
    with open(json_path,"r") as fl:
        labelMap=json.load(fl)
    return labelMap

labelMap = __loadLablMap__()
label_classes=list(labelMap.values())

class MaskRCNN_Mahindra:
    def __init__(self):
        global CONFIG_YAML_FL_PATH, MASK_MODEL_PATH, DETECTTHRESH, logger
        self.predictor = None
        self.mrcnn_config_fl = CONFIG_YAML_FL_PATH
        self.mrcnn_model_loc = MASK_MODEL_PATH
        self.mrcnn_model_fl = "model_final.pth"
        self.detection_thresh = DETECTTHRESH
        json_file = open(json_path)
        data = json.load(json_file)
        all_class_name = list(data.values())
        self.ALL_CLASS_NAMES = all_class_name
        self.NUMCLASSES = len(all_class_name)

        self.register_modeldatasets()
        logger.debug("Mask RCNN INIT Completed")
    
    def register_modeldatasets(self):
        global ALL_CLASS_NAMES, NUMCLASSES, DEMO_RUN_FLAG, logger
        tag = "mahindra_test"
        MetadataCatalog.get(tag).set(thing_classes=[label_classes])
        self.railway_metadata = MetadataCatalog.get(tag)
        cfg = get_cfg()
        cfg.merge_from_file(self.mrcnn_config_fl)
        if DEMO_RUN_FLAG:
            cfg.MODEL.DEVICE='cpu'
            
        cfg.MODEL.ROI_HEADS.NUM_CLASSES =self.NUMCLASSES
        cfg.OUTPUT_DIR = self.mrcnn_model_loc
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, self.mrcnn_model_fl)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_thresh
        self.predictor = DefaultPredictor(cfg)

    def run_inference_new(self,img_path):
        try:
            output = self.predictor(img_path)
            # output = self.predictor(img)
        except Exception as e:
            print(traceback.format_exc())
            logger.debug(f"run_inference_new() : Exception is : {e}, Traceback : {traceback.format_exc()}")

        predictions=output["instances"].to("cpu")
        boxes_surface = predictions.pred_boxes.tensor.to("cpu").numpy()
        pred_class_surface = predictions.pred_classes.to("cpu").numpy()
        #print(pred_class_surface.tolist())
        scores_surface = predictions.scores.to("cpu").numpy()  

        masklist = []
        masks = None  
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks.numpy() 
        #print("Only INF Time  = ", int(time.time()*1000) - total_t)     
        return self.processObjectList(boxes_surface,pred_class_surface, scores_surface,masks,img_path)

    def processObjectList(self, boxes_surface,pred_class_surface, scores_surface,masks,img_path):   
        global ALL_CLASS_NAMES, DETECTTHRESH, logger
        try:
            OBJECT_LIST = []
            
            # Step 1: Process detected objects
            for i, box in enumerate(boxes_surface):
                label_name = label_classes[pred_class_surface[i]]
                
                if scores_surface[i] > DETECTTHRESH:
                    xmin = int(box[0])
                    ymin = int(box[1])
                    xmax = int(box[2])
                    ymax = int(box[3])                
                    item = []
                    
                    # Skip specific classes
                    if label_name in ["NUT_MISSING", "NUT_PRESENT"]:
                        continue
                    if label_name == "HLA_TOP" and xmin < 200:
                        print(f"CLASS NAME IS {label_name} out of ROI")
                        continue

                    item.extend([label_name, xmin, ymin, xmax, ymax])

                    # Attempt to get mask list
                    try:
                        masklist = np.column_stack(np.where(masks[i] == True))
                    except Exception as ex:
                        print(f"Mask extraction error: {ex}")
                        masklist = []  

                    # Compute centroid
                    cx, cy = get_centroid(xmin, xmax, ymin, ymax)
                    item.extend([scores_surface[i], cx, cy, masklist])
                    
                    OBJECT_LIST.append(item)

            # Step 2: Find ROI region
            ROI_box = None
            filtered_objects = []

            for obj in OBJECT_LIST:
                if obj[0] == "ROI":  # Identify ROI region
                    _, ROI_xmin, ROI_ymin, ROI_xmax, ROI_ymax, _, cx, cy, mask = obj
                    ROI_box = (ROI_xmin, ROI_xmax, ROI_ymin, ROI_ymax)  # Maintain consistent order
                    break  # Stop after finding the ROI

            # Step 3: Filter objects inside ROI
            if ROI_box:
                ROI_xmin, ROI_xmax, ROI_ymin, ROI_ymax = ROI_box
                
                for obj in OBJECT_LIST:
                    obj_class, obj_xmin, obj_ymin, obj_xmax, obj_ymax, _, cx, cy, mask = obj

                    # Check if the centroid is inside the ROI
                    if ROI_xmin <= cx <= ROI_xmax and ROI_ymin <= cy <= ROI_ymax:
                        filtered_objects.append(obj)

                        # Draw box only for filtered objects
                        drawCV2Box(img_path, obj_class, obj_xmin, obj_ymin, obj_xmax, obj_ymax)
                    else:
                        print("Out of range:", obj)

            # Step 4: Print filtered objects
          #  print("Filtered Objects:", filtered_objects)

        except Exception as e:
            logger.debug(f"Exception in processObjectList: {e}")
            logger.critical(f"processObjectList() Exception: {e}, Traceback: {traceback.format_exc()}")
            return img_path, filtered_objects

        return img_path, filtered_objects
    
    

def get_centroid(xmin, xmax, ymin, ymax):
    cx = int((xmin + xmax) / 2.0)
    cy = int((ymin + ymax) / 2.0)
    return(cx, cy)

def drawCV2Box(frame,labelName, xmin,ymin,xmax,ymax):
    try: 
        if labelName == "NUMBER_ROI":
            labelName = '' 
        if labelName == "CAM_CAP_OK_MISSING" or labelName == "HLA_BASE"  or labelName == "HALF_HLA_BASE":  
            cv2.putText(frame, labelName, (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 0, 255), 3, cv2.LINE_AA) 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 7)
        else:         
            cv2.putText(frame, labelName, (xmin,ymin-20), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 255, 0), 3, cv2.LINE_AA)        
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0),4)
    except Exception as e:
        print("Exception in drawCV2Box() : "+ str(e))


def getInferenceTrigger():
    Engine_no = None
    Engine_type = None
    try:
        # Database connection
        db_fetch = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        with db_fetch.cursor() as cur:
            # Fetch only the latest entry
            query = "SELECT ENGINE_NUMBER, ENGINE_TYPE FROM engine_number_table ORDER BY ID DESC LIMIT 1;"
            cur.execute(query)
            data = cur.fetchone()
            
            if data:
                Engine_no, Engine_type = str(data[0]), str(data[1])

    except Exception as e:
        print(f"Database Error: {e}")
        logger.debug(f"getInferenceTrigger expection is : {e}")

    finally:
        if db_fetch:
            db_fetch.close()

    return Engine_no, Engine_type


def insertDataIndotprocessing_table(Engine_no,EngineTypeProcess, OVREALL_STATUS, POS_1_STATUS, POS_2_STATUS, POS_3_STATUS, POS_4_STATUS, POS_5_STATUS, POS_6_STATUS, POS_7_STATUS, POS_8_STATUS,
                                                                                                  CAM1_IMAGE_LINK, CAM2_IMAGE_LINK,CAM3_IMAGE_LINK,CAM4_IMAGE_LINK,CAM5_IMAGE_LINK,CAM6_IMAGE_LINK,CAM7_IMAGE_LINK,CAM8_IMAGE_LINK):
    IS_PROCESS_INF = 0
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_update.cursor()
        query = "INSERT INTO cam_cap_processing_table (ENIGNE_NUMBER, ENGINE_TYPE, OVERALL_STATUS, POS_1_STATUS,POS_2_STATUS,POS_3_STATUS,POS_4_STATUS,POS_5_STATUS,POS_6_STATUS,POS_7_STATUS,POS_8_STATUS,POS_1_IMAGE_LINK,POS_2_IMAGE_LINK,POS_3_IMAGE_LINK,POS_4_IMAGE_LINK,POS_5_IMAGE_LINK,POS_6_IMAGE_LINK,POS_7_IMAGE_LINK,POS_8_IMAGE_LINK) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        cur.execute(query, (Engine_no, EngineTypeProcess, OVREALL_STATUS, POS_1_STATUS, POS_2_STATUS,POS_3_STATUS,POS_4_STATUS,POS_5_STATUS,POS_6_STATUS,POS_7_STATUS,POS_8_STATUS,CAM1_IMAGE_LINK,CAM2_IMAGE_LINK,CAM3_IMAGE_LINK,CAM4_IMAGE_LINK,CAM5_IMAGE_LINK,CAM6_IMAGE_LINK,CAM7_IMAGE_LINK,CAM8_IMAGE_LINK))
        db_update.commit()
        IS_PROCESS_INF = 1
    except Exception as e:
        print(e)
        logger.debug(f"except Exception insertDataIndotprocessing_table {e}")
        logger.error(f"Error in inserting data: {e}, Traceback : {traceback.format_exc()}")
    finally:
        if cur:
            cur.close()
        if db_update:
            db_update.close()

    return IS_PROCESS_INF


def insert_CamCap_HLA_Status_table_4CYLINDER(Engine_no, EngineTypeProcess, OVERALL_CAM_CAP_RESULT, CAM_CAP_NUMBER_STATUS_POS1, CAM_CAP_NUMBER_STATUS_POS2, CAM_CAP_NUMBER_STATUS_POS3, CAM_CAP_NUMBER_STATUS_POS4, CAM_CAP_NUMBER_STATUS_POS5, CAM_CAP_NUMBER_STATUS_POS6, CAM_CAP_NUMBER_STATUS_POS7, CAM_CAP_NUMBER_STATUS_POS8,
                                   OVERALL_HLA_RESULT, HLA_PostionCheck_pos1, HLA_PostionCheck_pos2, HLA_PostionCheck_pos3, HLA_PostionCheck_pos4, HLA_PostionCheck_pos5, HLA_PostionCheck_pos6, HLA_PostionCheck_pos7, HLA_PostionCheck_pos8,pdfFIle_Name):
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_update.cursor()
        query = """INSERT INTO cam_cap_hla_status_table 
                   (ENIGNE_NUMBER, ENGINE_TYPE, CAM_CAP_OVERALL_STATUS, I2, I3, I4, I5, E1, E2, E3, E4, HLA_OVERALL_STATUS, 
                    I2_HLA, I3_HLA, I4_HLA, I5_HLA, E1_HLA, E2_HLA, E3_HLA, E4_HLA,PDF_FILE_PATH) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
        cur.execute(query, (Engine_no, EngineTypeProcess, OVERALL_CAM_CAP_RESULT, 
                            CAM_CAP_NUMBER_STATUS_POS1, CAM_CAP_NUMBER_STATUS_POS2, CAM_CAP_NUMBER_STATUS_POS3, CAM_CAP_NUMBER_STATUS_POS4, 
                            CAM_CAP_NUMBER_STATUS_POS5, CAM_CAP_NUMBER_STATUS_POS6, CAM_CAP_NUMBER_STATUS_POS7, CAM_CAP_NUMBER_STATUS_POS8, 
                            OVERALL_HLA_RESULT, HLA_PostionCheck_pos1, HLA_PostionCheck_pos2, HLA_PostionCheck_pos3, HLA_PostionCheck_pos4, 
                            HLA_PostionCheck_pos5, HLA_PostionCheck_pos6, HLA_PostionCheck_pos7, HLA_PostionCheck_pos8,pdfFIle_Name))
        db_update.commit()
    except Exception as e:
        print(f"Error: {e}")
        logger.debug(f"Exception in insert_CamCap_HLA_Status_table_4CYLINDER: {e}")
        logger.error(f"Error in insert_CamCap_HLA_Status_table_4CYLINDER: {e}, Traceback: {traceback.format_exc()}")
    finally:
        if cur:
            cur.close()
        if db_update:
            db_update.close()


def insert_CamCap_HLA_Status_table_3CYLINDER(Engine_no, EngineTypeProcess, OVERALL_CAM_CAP_RESULT, CAM_CAP_NUMBER_STATUS_POS1, CAM_CAP_NUMBER_STATUS_POS2, CAM_CAP_NUMBER_STATUS_POS3, CAM_CAP_NUMBER_STATUS_POS5, CAM_CAP_NUMBER_STATUS_POS6, CAM_CAP_NUMBER_STATUS_POS7,
                                   OVERALL_HLA_RESULT, HLA_PostionCheck_pos1, HLA_PostionCheck_pos2, HLA_PostionCheck_pos3, HLA_PostionCheck_pos5, HLA_PostionCheck_pos6, HLA_PostionCheck_pos7,pdfFIle_Name):
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_update.cursor()
        query = """INSERT INTO cam_cap_hla_status_table 
                   (ENIGNE_NUMBER, ENGINE_TYPE, CAM_CAP_OVERALL_STATUS, I1, I2, I3,  E1, E2, E3,  HLA_OVERALL_STATUS, 
                    I1_HLA, I2_HLA, I3_HLA, E1_HLA, E2_HLA, E3_HLA,PDF_FILE_PATH) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
        cur.execute(query, (Engine_no, EngineTypeProcess, OVERALL_CAM_CAP_RESULT, 
                            CAM_CAP_NUMBER_STATUS_POS1, CAM_CAP_NUMBER_STATUS_POS2, CAM_CAP_NUMBER_STATUS_POS3, 
                            CAM_CAP_NUMBER_STATUS_POS5, CAM_CAP_NUMBER_STATUS_POS6, CAM_CAP_NUMBER_STATUS_POS7, 
                            OVERALL_HLA_RESULT, HLA_PostionCheck_pos1, HLA_PostionCheck_pos2, HLA_PostionCheck_pos3, 
                            HLA_PostionCheck_pos5, HLA_PostionCheck_pos6, HLA_PostionCheck_pos7, pdfFIle_Name))
        db_update.commit()
    except Exception as e:
        print(f"Error: {e}")
        logger.debug(f"Exception in insert_CamCap_HLA_Status_table_3CYLINDER: {e}")
        logger.error(f"Error in insert_CamCap_HLA_Status_table_3CYLINDER: {e}, Traceback: {traceback.format_exc()}")
    finally:
        if cur:
            cur.close()
        if db_update:
            db_update.close()

def insertDataIndotprocessing_table_3cy(Engine_no, EngineTypeProcess, processDateTime, OVREALL_STATUS, POS_1_STATUS, POS_2_STATUS, POS_3_STATUS, POS_4_STATUS, POS_5_STATUS, POS_6_STATUS,
                                                                                                  CAM1_IMAGE_LINK, CAM2_IMAGE_LINK,CAM3_IMAGE_LINK,CAM4_IMAGE_LINK,CAM5_IMAGE_LINK,CAM6_IMAGE_LINK):
    IS_PROCESS_INF = 0
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_update.cursor()
        query = "INSERT INTO cam_cap_processing_table_3Cylinder (ENIGNE_NUMBER, ENGINE_TYPE, DATE_TIME, OVERALL_STATUS, POS_1_STATUS,POS_2_STATUS,POS_3_STATUS,POS_4_STATUS,POS_5_STATUS,POS_6_STATUS,POS_1_IMAGE_LINK,POS_2_IMAGE_LINK,POS_3_IMAGE_LINK,POS_4_IMAGE_LINK,POS_5_IMAGE_LINK,POS_6_IMAGE_LINK) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        cur.execute(query, (Engine_no, EngineTypeProcess, processDateTime, OVREALL_STATUS, POS_1_STATUS, POS_2_STATUS,POS_3_STATUS,POS_4_STATUS,POS_5_STATUS,POS_6_STATUS,CAM1_IMAGE_LINK,CAM2_IMAGE_LINK,CAM3_IMAGE_LINK,CAM4_IMAGE_LINK,CAM5_IMAGE_LINK,CAM6_IMAGE_LINK))
        db_update.commit()
        IS_PROCESS_INF = 1
    except Exception as e:
        logger.debug(f"except Exception insertDataIndotprocessing_table_3cy {e}")
        logger.error(f"Error in inserting insertDataIndotprocessing_table_3cy: {e}, Traceback : {traceback.format_exc()}")

    finally:
        if cur:
            cur.close()
        if db_update:
            db_update.close()

    return IS_PROCESS_INF

#OVREALL_STATUS
def overAllStatus_update_ForPLC(ENGINE_NUMBER, OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT, IS_PROCESS_INF):
    try:
        db_insert = pymysql.connect(host=DB_HOST,user=DB_USER, passwd=DB_PASS,db= DB_NAME)
        cur = db_insert.cursor()

        query = "UPDATE engine_status_config SET ENGINE_NUMBER = %s, OVREALL_STATUS = %s, OVERALL_CAM_CAP_RESULT = %s, OVERALL_HLA_RESULT = %s, IS_PROCESS_INF = %s WHERE ID = 1"
        cur.execute(query, (ENGINE_NUMBER,OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT,IS_PROCESS_INF))
        db_insert.commit()
    except Exception as e:
        print("Exception in overAllStatus_update_ForPLC",e)
        logger.critical("overAllStatus_update_ForPLC Exception is : "+str(e))
    finally:
        if cur:
            cur.close()
        if db_insert:
            db_insert.close()

def update_vision_overall_Status(STATUS):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db= DB_NAME)
        cur = db_update.cursor()
        query = "UPDATE status_config_table SET STATUS = %s WHERE ID = 1"
        cur.execute(query, (STATUS))
        db_update.commit()
    except Exception as e:
        print("update_PLC_Status_UI() Exception is : "+ str(e))
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close()


def HLA_classCheck(OBJECT_LIST):
    TOP_HLA_CLASS_NAME_LIST1_STATUS = False
    try:
        if sum(sublist.count('HLA_TOP') for sublist in OBJECT_LIST) == 2:
            print(" HLA_classCheck Status OK ")
            TOP_HLA_CLASS_NAME_LIST1_STATUS = True    
            #print("TOP_HLA 2 IS NOT OK")  

        else:
            print(" HLA_classCheck Status not OK ")
            TOP_HLA_CLASS_NAME_LIST1_STATUS = False

        return TOP_HLA_CLASS_NAME_LIST1_STATUS
    except Exception as e:
        logger.debug(f"except Exception HLA_classCheck {e}")
        print("HLA_classCheck is :",e)

def putText_notok_ok(textMsg,frame, pos):
    try:
        if textMsg == "CamCap is NotOK" or textMsg == "HLA is NotOK":
            position = pos
            text = textMsg
            font_scale = 2
            #color = (255, 255, 255)
                # if isENGINEOk is True:
            color = (0, 0, 255)#(0, 255, 0)
            thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_type = cv2.LINE_AA
        
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            line_height = text_size[1] + 25
            x, y0 = position
            for i, line in enumerate(text.split("\n")):
                y = y0 + i * line_height
                cv2.putText(frame,
                            line,
                            (x, y),
                            font,
                            font_scale,
                            color,
                            thickness,
                            line_type)
        else:
            position = pos
            text = textMsg
            font_scale = 2
            #color = (255, 255, 255)
                # if isENGINEOk is True:
            color =(0, 255, 0)
            thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_type = cv2.LINE_AA
        
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            line_height = text_size[1] + 25
            x, y0 = position
            for i, line in enumerate(text.split("\n")):
                y = y0 + i * line_height
                cv2.putText(frame,
                            line,
                            (x, y),
                            font,
                            font_scale,
                            color,
                            thickness,
                            line_type)   
    except Exception as e:
        print(e)




def is_inside_number_roi(obj_box, number_roi_box):
    x1, y1, x2, y2 = obj_box
    rx1, ry1, rx2, ry2 = number_roi_box
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2



def HLA_CAM_CAP_INSPECTION_4CYLINDER(position, OBJECT_LIST, original_image, type):
    try:
        # Extract NUMBER_ROI bounding box
        number_roi_box = None
        for obj in OBJECT_LIST:
            if obj[0] == "NUMBER_ROI":
                number_roi_box = (obj[1], obj[2], obj[3], obj[4])
                break

        if number_roi_box is None:
            print("NUMBER_ROI not found")
            return "NOT OK", original_image

        if type == 1:
            status_mapping = {
                "1": "CAM_CAP_I1_STATUS",
                "2": "CAM_CAP_I2_STATUS",
                "3": "CAM_CAP_I3_STATUS",
                "4": "CAM_CAP_I4_STATUS",
                "5": "CAM_CAP_E4_STATUS",
                "6": "CAM_CAP_E3_STATUS",
                "7": "CAM_CAP_E2_STATUS",
                "8": "CAM_CAP_E1_STATUS",
            }
            status = {key: "NOT OK" for key in status_mapping.values()}
            object_flags = {
                "I": False, "E": False,
                "Ione": False, "Itwo": False, "Ithree": False, "Ifour": False,
                "Eone": False, "Etwo": False, "Ethree": False, "Efour": False
            }

            if position in status_mapping:
                for obj in OBJECT_LIST:
                    classname = obj[0]
                    if classname != "NUMBER_ROI" and not is_inside_number_roi((obj[1], obj[2], obj[3], obj[4]), number_roi_box):
                        continue

                    if classname == "I" and position in ["1", "2", "3", "4"]:
                        object_flags["I"] = True
                    elif classname == "E" and position in ["5", "6", "7", "8"]:
                        object_flags["E"] = True
                    elif classname == "1" and position == "1":
                        object_flags["Ione"] = True
                    elif classname == "2" and position == "2":
                        object_flags["Itwo"] = True
                    elif classname == "3" and position == "3":
                        object_flags["Ithree"] = True
                    elif classname == "4" and position == "4":
                        object_flags["Ifour"] = True
                    elif classname == "4" and position == "5":
                        object_flags["Efour"] = True
                    elif classname == "3" and position == "6":
                        object_flags["Ethree"] = True
                    elif classname == "2" and position == "7":
                        object_flags["Etwo"] = True
                    elif classname == "1" and position == "8":
                        object_flags["Eone"] = True
                    elif classname not in ["NUMBER_ROI", "HLA_ROD_TOP", "CAM_CAP_OK", "HLA_TOP", "NUT_PRESENT", "HLA_3CYL", "ROI", "HLA_3CYL_LINE"]:
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (obj[1], obj[2]), (obj[3], obj[4]), color, 4)
                        cv2.putText(original_image, classname, (obj[1], obj[2] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

                        object_flags = {k: False for k in object_flags}

                # Status assignment
                if position == "1" and object_flags["I"] and object_flags["Ione"]:
                    status["CAM_CAP_I1_STATUS"] = "OK"
                elif position == "2" and object_flags["I"] and object_flags["Itwo"]:
                    status["CAM_CAP_I2_STATUS"] = "OK"
                elif position == "3" and object_flags["I"] and object_flags["Ithree"]:
                    status["CAM_CAP_I3_STATUS"] = "OK"
                elif position == "4" and object_flags["I"] and object_flags["Ifour"]:
                    status["CAM_CAP_I4_STATUS"] = "OK"
                elif position == "5" and object_flags["E"] and object_flags["Efour"]:
                    status["CAM_CAP_E4_STATUS"] = "OK"
                elif position == "6" and object_flags["E"] and object_flags["Ethree"]:
                    status["CAM_CAP_E3_STATUS"] = "OK"
                elif position == "7" and object_flags["E"] and object_flags["Etwo"]:
                    status["CAM_CAP_E2_STATUS"] = "OK"
                elif position == "8" and object_flags["E"] and object_flags["Eone"]:
                    status["CAM_CAP_E1_STATUS"] = "OK"

                return status[status_mapping[position]], original_image

        else:
            status_mapping = {
                "1": "CAM_CAP_I2_STATUS",
                "2": "CAM_CAP_I3_STATUS",
                "3": "CAM_CAP_I4_STATUS",
                "4": "CAM_CAP_I5_STATUS",
                "5": "CAM_CAP_E4_STATUS",
                "6": "CAM_CAP_E3_STATUS",
                "7": "CAM_CAP_E2_STATUS",
                "8": "CAM_CAP_E1_STATUS",
            }
            status = {key: "NOT OK" for key in status_mapping.values()}
            object_flags = {
                "I": False, "E": False,
                "Itwo": False, "Ithree": False, "Ifour": False, "Ifive": False,
                "Eone": False, "Etwo": False, "Ethree": False, "Efour": False
            }

            if position in status_mapping:
                for obj in OBJECT_LIST:
                    classname = obj[0]
                    if classname != "NUMBER_ROI" and not is_inside_number_roi((obj[1], obj[2], obj[3], obj[4]), number_roi_box):
                        continue

                    if classname == "I" and position in ["1", "2", "3", "4"]:
                        object_flags["I"] = True
                    elif classname == "E" and position in ["5", "6", "7", "8"]:
                        object_flags["E"] = True
                    elif classname == "2" and position == "1":
                        object_flags["Itwo"] = True
                    elif classname == "3" and position == "2":
                        object_flags["Ithree"] = True
                    elif classname == "4" and position == "3":
                        object_flags["Ifour"] = True
                    elif classname == "5" and position == "4":
                        object_flags["Ifive"] = True
                    elif classname == "4" and position == "5":
                        object_flags["Efour"] = True
                    elif classname == "3" and position == "6":
                        object_flags["Ethree"] = True
                    elif classname == "2" and position == "7":
                        object_flags["Etwo"] = True
                    elif classname == "1" and position == "8":
                        object_flags["Eone"] = True
                    elif classname not in ["NUMBER_ROI", "HLA_ROD_TOP", "CAM_CAP_OK", "HLA_TOP", "NUT_PRESENT", "HLA_3CYL", "ROI", "HLA_3CYL_LINE"]:
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (obj[1], obj[2]), (obj[3], obj[4]), color, 4)
                        cv2.putText(original_image, classname, (obj[1], obj[2] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

                        object_flags = {k: False for k in object_flags}

                # Status assignment
                if position == "1" and object_flags["I"] and object_flags["Itwo"]:
                    status["CAM_CAP_I2_STATUS"] = "OK"
                elif position == "2" and object_flags["I"] and object_flags["Ithree"]:
                    status["CAM_CAP_I3_STATUS"] = "OK"
                elif position == "3" and object_flags["I"] and object_flags["Ifour"]:
                    status["CAM_CAP_I4_STATUS"] = "OK"
                elif position == "4" and object_flags["I"] and object_flags["Ifive"]:
                    status["CAM_CAP_I5_STATUS"] = "OK"
                elif position == "5" and object_flags["E"] and object_flags["Efour"]:
                    status["CAM_CAP_E4_STATUS"] = "OK"
                elif position == "6" and object_flags["E"] and object_flags["Ethree"]:
                    status["CAM_CAP_E3_STATUS"] = "OK"
                elif position == "7" and object_flags["E"] and object_flags["Etwo"]:
                    status["CAM_CAP_E2_STATUS"] = "OK"
                elif position == "8" and object_flags["E"] and object_flags["Eone"]:
                    status["CAM_CAP_E1_STATUS"] = "OK"

                return status[status_mapping[position]], original_image

    except Exception as e:
        print(f"Exception in HLA_CAM_CAP_INSPECTION_4CYLINDER: {e}")
        logger.debug(f"Exception in HLA_CAM_CAP_INSPECTION_4CYLINDER: {e}")
        return "NOT OK", original_image



def HLA_CAM_CAP_INSPECTION_3CYLINDER(position, OBJECT_LIST, original_image):
    import cv2  # Ensure cv2 is imported

    # Mapping of position to status variable names
    status_mapping = {
        "1": "CAM_CAP_I1_STATUS",
        "2": "CAM_CAP_I2_STATUS",
        "3": "CAM_CAP_I3_STATUS",
        "4": "CAM_CAP_E1_STATUS",
        "5": "CAM_CAP_E2_STATUS",
        "6": "CAM_CAP_E3_STATUS"
    }

    # Initialize all statuses to "NOT OK"
    status = {key: "NOT OK" for key in status_mapping.values()}

    # Flags for different object types and specific positions
    object_flags = {
        "I": False, "E": False,
        "Ione": False, "Itwo": False, "Ithree": False,
        "Eone": False, "Etwo": False, "Ethree": False
    }

    try:
        # Find NUMBER_ROI bounding box
        roi_box = None
        for obj in OBJECT_LIST:
            if obj[0] == "NUMBER_ROI":
                roi_box = obj[1], obj[2], obj[3], obj[4]  # x1, y1, x2, y2
                break

        if not roi_box:
            print("⚠️ NUMBER_ROI not found")
            return "NOT OK", original_image

        x1_roi, y1_roi, x2_roi, y2_roi = roi_box

        # Filter objects inside ROI
        filtered_objects = []
        for obj in OBJECT_LIST:
            print("obj is :",obj)
            class_name, x1, y1, x2, y2 = obj[:5]
            # Check if object's center is inside ROI
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if x1_roi <= cx <= x2_roi and y1_roi <= cy <= y2_roi:
                filtered_objects.append(obj)

        # Evaluate objects inside ROI
        if position in status_mapping:
            for obj in filtered_objects:
                classname = obj[0]

                # Set flags
                if classname == "I" and position in ["1", "2", "3"]:
                    object_flags["I"] = True
                elif classname == "E" and position in ["4", "5", "6"]:
                    object_flags["E"] = True
                elif classname == "1" and position == "1":
                    object_flags["Ione"] = True
                elif classname == "2" and position == "2":
                    object_flags["Itwo"] = True
                elif classname == "3" and position == "3":
                    object_flags["Ithree"] = True
                elif classname == "1" and position == "6":
                    object_flags["Eone"] = True
                elif classname == "2" and position == "5":
                    object_flags["Etwo"] = True
                elif classname == "3" and position == "4":
                    object_flags["Ethree"] = True

                # Draw red box for unexpected classes
                elif classname not in [
                    "NUMBER_ROI", "HLA_ROD_TOP", "CAM_CAP_OK", "HLA_TOP",
                    "NUT_PRESENT", "HLA_3CYL", "ROI", "HLA_3CYL_LINE"
                ]:
                    print(f"⚠️ Unexpected class: {classname}")
                    cv2.rectangle(original_image, (obj[1], obj[2]), (obj[3], obj[4]), (0, 0, 255), 3)

        # Final decision logic based on position
        if position == "1" and object_flags["I"] and object_flags["Ione"]:
            status["CAM_CAP_I1_STATUS"] = "OK"
        elif position == "2" and object_flags["I"] and object_flags["Itwo"]:
            status["CAM_CAP_I2_STATUS"] = "OK"
        elif position == "3" and object_flags["I"] and object_flags["Ithree"]:
            status["CAM_CAP_I3_STATUS"] = "OK"
        elif position == "4" and object_flags["E"] and object_flags["Ethree"]:
            status["CAM_CAP_E1_STATUS"] = "OK"
        elif position == "5" and object_flags["E"] and object_flags["Etwo"]:
            status["CAM_CAP_E2_STATUS"] = "OK"
        elif position == "6" and object_flags["E"] and object_flags["Eone"]:
            status["CAM_CAP_E3_STATUS"] = "OK"

        return status[status_mapping[position]], original_image

    except Exception as e:
        print(f"❌ Exception: {e}")
        return "NOT OK", original_image





def save_cam_image_for_mt_list(position, status_text, position_status_var, image_suffix, engine_folder_path, original_image, imreal):
    try:
        # Annotate the image
        pos = (50, 60)
        putText_notok_ok(status_text, original_image, pos)
        # Set position status
        globals()[position_status_var] = "NOT OK"

        # Get the current time for the filename
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Generate image paths including the folder structure
        real_image_path = os.path.join(engine_folder_path, f"{formatted_datetime}_{image_suffix}_real.jpg")
        annotated_image_path = os.path.join(engine_folder_path, f"{formatted_datetime}_{image_suffix}.jpg")

        # Save images
        cv2.imwrite(real_image_path, imreal)
        cv2.imwrite(annotated_image_path, original_image)

        return annotated_image_path
    
    except Exception as e:
        logger.debug(f"Exception in save_cam_image_for_mt_list: {e}")
        print("Exception in save_cam_image_for_mt_list",e)

def save_cam_image_for_mt_list_3cylinder(position, status_text, position_status_var, image_suffix, engine_folder_path, original_image, imreal):
    try:
        # Annotate the image
        pos = (50, 60)
        putText_notok_ok(status_text, original_image, pos)
        # Set position status
        globals()[position_status_var] = "NOT OK"

        # Get the current time for the filename
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Generate image paths including the folder structure
        real_image_path = os.path.join(engine_folder_path, f"{formatted_datetime}_{image_suffix}_real.jpg")
        annotated_image_path = os.path.join(engine_folder_path, f"{formatted_datetime}_{image_suffix}.jpg")

        # Save images
        cv2.imwrite(real_image_path, imreal)
        cv2.imwrite(annotated_image_path, original_image)

        return annotated_image_path
    
    except Exception as e:
        logger.debug(f"Exception in save_cam_image_for_mt_list: {e}")
        print("Exception in save_cam_image_for_mt_list",e)


def HLA_PositionCheck_WITH_ROD_4CYLINDER(Position, OBJECT_LIST, original_image):
    # Initialize statuses and variables
    HLA_POSTION_RIGHT = "NOT OK"
    HLA_POSTION_LEFT = "NOT OK"
    HLA_POSTION = "NOT OK"

    thresholdPixel_LeftSide_1 = 70
    thresholdPixel_LeftSide_2 = 20

    thresholdPixel_RightSide_1 = 70
    thresholdPixel_RightSide_2 = 40


    try:
        if Position == "1":  # I2 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 605: # 500 # Left-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    #elif smalllabellist[0] == 'HLA_ROD_TOP':
                    elif smalllabellist[0] == 'HLA_ROD_TOP' and ymax > 250: #60
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                   # elif smalllabellist[0] == 'HLA_ROD_TOP':
                    elif smalllabellist[0] == 'HLA_ROD_TOP' and ymin < 1050:
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    thresholdPixel_LeftSide_1 = 56#51#41#36#25#14
                    thresholdPixel_LeftSide_2 = 89#81#75#65#50
                    LeftSide_Cet = Lcy - cy   #ok -16

                    logger.debug(f"Left I2, LeftSide_Cet_4cy is {LeftSide_Cet}, yminTL {yminTL} ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, yminRTL: {yminRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                   
                    #Left I2, LeftSide_Cet_4cy is -284, yminTL 279 ymaxTL: 578, ymaxRTL: 247, yminRTL: 42, thresholdPixel_LeftSide_1: 41, thresholdPixel_LeftSide_2: 75
                    #Left I2, LeftSide_Cet_4cy is -64, yminTL 241 ymaxTL: 528, ymaxRTL: 451, yminRTL: 189, thresholdPixel_LeftSide_1: 41, thresholdPixel_LeftSide_2: 75
                    #Left I2, LeftSide_Cet_4cy is -70, yminTL 202 ymaxTL: 490, ymaxRTL: 406, yminRTL: 147, thresholdPixel_LeftSide_1: 41, thresholdPixel_LeftSide_2: 81
                    #Left I2, LeftSide_Cet_4cy is 31, yminTL 277 ymaxTL: 578, ymaxRTL: 593, yminRTL: 324, thresholdPixel_LeftSide_1: 41, thresholdPixel_LeftSide_2: 89
                    #Left I2, LeftSide_Cet_4cy is -297, yminTL 290 ymaxTL: 589, ymaxRTL: 244, yminRTL: 41, thresholdPixel_LeftSide_1: 51, thresholdPixel_LeftSide_2: 89
                    #Left I2, LeftSide_Cet_4cy is 31, yminTL 268 ymaxTL: 572, ymaxRTL: 581, yminRTL: 321, thresholdPixel_LeftSide_1: 51, thresholdPixel_LeftSide_2: 89
                    #Left I2, LeftSide_Cet_4cy is -288, yminTL 303 ymaxTL: 602, ymaxRTL: 267, yminRTL: 61, thresholdPixel_LeftSide_1: 56, thresholdPixel_LeftSide_2: 89


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 160), (1600, 515), color, 12)
                    logger.debug("I2 left HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"I2 Exception in Left Side check: {e}")
                print("I2 Exception in Left Side check:", e)

            # Right Side check
            try:              
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 82#75#67#61#57#45#35
                    thresholdPixel_RightSide_2 = 60#52#45#38
                    HlaSize = ymaxTR -yminTR
                    ymin_R_Value = yminTR -yminRTR
                    ymax_R_Value = ymaxTR -ymaxRTR
                    RightSide_Cet = cy -Rcy #ok 27 ok 54
                   
                    logger.debug(f"Right I2,ymax_R_Value is {ymax_R_Value} RightSide_Cet is {RightSide_Cet} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right I2,ymax_R_Value is -17 RightSide_Cet is -24 yminTR is 818 yminRTR 850, ymaxTR: 1114, ymaxRTR: 1131, thresholdPixel_RightSide_1: 35, thresholdPixel_RightSide_2: 30
                    #Right I2,ymax_R_Value is -34 RightSide_Cet is -37 yminTR is 640 yminRTR 681, ymaxTR: 898, ymaxRTR: 932, thresholdPixel_RightSide_1: 35, thresholdPixel_RightSide_2: 38
                    #Right I2,ymax_R_Value is 39 RightSide_Cet is 30 yminTR is 805 yminRTR 784, ymaxTR: 1103, ymaxRTR: 1064, thresholdPixel_RightSide_1: 35, thresholdPixel_RightSide_2: 45
                    #Right I2,ymax_R_Value is -41 RightSide_Cet is -45 yminTR is 624 yminRTR 672, ymaxTR: 885, ymaxRTR: 926, thresholdPixel_RightSide_1: 35, thresholdPixel_RightSide_2: 45
                    #Right I2,ymax_R_Value is -50 RightSide_Cet is -49 yminTR is 640 yminRTR 687, ymaxTR: 893, ymaxRTR: 943, thresholdPixel_RightSide_1: 35, thresholdPixel_RightSide_2: 45
                    #Right I2,ymax_R_Value is 63 RightSide_Cet is 48 yminTR is 759 yminRTR 727, ymaxTR: 1063, ymaxRTR: 1000, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 52
                    #Right I2,ymax_R_Value is 63 RightSide_Cet is 48 yminTR is 759 yminRTR 727, ymaxTR: 1063, ymaxRTR: 1000, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 52
                    #Right I2,ymax_R_Value is -45 RightSide_Cet is -51 yminTR is 629 yminRTR 685, ymaxTR: 884, ymaxRTR: 929, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 52
                    #Right I2,ymax_R_Value is 52 RightSide_Cet is 40 yminTR is 720 yminRTR 692, ymaxTR: 1017, ymaxRTR: 965, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 60
                    #Right I2,ymax_R_Value is 87 RightSide_Cet is 72 yminTR is 705 yminRTR 649, ymaxTR: 1015, ymaxRTR: 928, thresholdPixel_RightSide_1: 57, thresholdPixel_RightSide_2: 60
                    #Right I2,ymax_R_Value is 58 RightSide_Cet is 48 yminTR is 808 yminRTR 770, ymaxTR: 1110, ymaxRTR: 1052, thresholdPixel_RightSide_1: 57, thresholdPixel_RightSide_2: 60

                    #Right I2,ymax_R_Value is 66 RightSide_Cet is 54 yminTR is 782 yminRTR 741, ymaxTR: 1090, ymaxRTR: 1024, thresholdPixel_RightSide_1: 61, thresholdPixel_RightSide_2: 60
                    #Right I2,ymax_R_Value is 68 RightSide_Cet is 49 yminTR is 793 yminRTR 762, ymaxTR: 1100, ymaxRTR: 1032, thresholdPixel_RightSide_1: 67, thresholdPixel_RightSide_2: 60
                    #Right I2,ymax_R_Value is 78 RightSide_Cet is 63 yminTR is 751 yminRTR 703, ymaxTR: 1056, ymaxRTR: 978, thresholdPixel_RightSide_1: 75, thresholdPixel_RightSide_2: 60


                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_1) : 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                      
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 760), (1600, 1110), color, 12)
                    logger.debug("I2 right HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I2 Exception in Right Side check: {e}")
                print("I2 Exception in Right Side check:", e)



        #==========================I3====================================
        if Position == "2":  # I3 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500:
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]
                    thresholdPixel_LeftSide_1 = 45#35#30#27#22#15#10#1
                    thresholdPixel_LeftSide_2 = 149#140#130#110#101#95
                    HlaSize = ymaxTL -yminTL
                    ymin_L_Value = yminTL -yminRTL
                    ymax_L_Value = ymaxTL -ymaxRTL
                    LeftSide_Cet = Lcy - cy   # not ok -19

                   
                    logger.debug(f"Left I3, LeftSide_Cet is {LeftSide_Cet} ymaxTL: {ymaxTL}, yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left I3, LeftSide_Cet is -46 ymaxTL: 306, ymaxRTL: 225, xminTL: 818, thresholdPixel_LeftSide_1: 0, thresholdPixel_LeftSide_2: 80
                    #Left I3, LeftSide_Cet is -30 ymaxTL: 460, yminTL 172 yminRTL 182 ymaxRTL: 390, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -61 ymaxTL: 391, yminTL 107 yminRTL 83 ymaxRTL: 293, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -31 ymaxTL: 477, yminTL 196 yminRTL 201 ymaxRTL: 409, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -32 ymaxTL: 479, yminTL 199 yminRTL 202 ymaxRTL: 413, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -28 ymaxTL: 486, yminTL 199 yminRTL 210 ymaxRTL: 419, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -22 ymaxTL: 498, yminTL 215 yminRTL 226 ymaxRTL: 442, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -19 ymaxTL: 490, yminTL 203 yminRTL 222 ymaxRTL: 433, thresholdPixel_LeftSide_1: 15, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -63 ymaxTL: 456, yminTL 172 yminRTL 145 ymaxRTL: 357, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 95
                    #Left I3, LeftSide_Cet is -63 ymaxTL: 469, yminTL 184 yminRTL 160 ymaxRTL: 366, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 101
                    #Left I3, LeftSide_Cet is -90 ymaxTL: 454, yminTL 174 yminRTL 117 ymaxRTL: 332, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 110
                    #Left I3, LeftSide_Cet is -96 ymaxTL: 372, yminTL 84 yminRTL 28 ymaxRTL: 236, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 130
                    #Left I3, LeftSide_Cet is -14 ymaxTL: 417, yminTL 126 yminRTL 149 ymaxRTL: 366, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 140
                    #Left I3, LeftSide_Cet is 0 ymaxTL: 501, yminTL 210 yminRTL 246 ymaxRTL: 465, thresholdPixel_LeftSide_1: 27, thresholdPixel_LeftSide_2: 140
                    #Left I3, LeftSide_Cet is 0 ymaxTL: 479, yminTL 186 yminRTL 226 ymaxRTL: 438, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 140
                    #Left I3, LeftSide_Cet is 0 ymaxTL: 479, yminTL 186 yminRTL 226 ymaxRTL: 438, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 140
                    #Left I3, LeftSide_Cet is -99 ymaxTL: 483, yminTL 192 yminRTL 134 ymaxRTL: 342, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 140
                    #Left I3, LeftSide_Cet is -104 ymaxTL: 472, yminTL 183 yminRTL 118 ymaxRTL: 329, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 140
                    #Left I3, LeftSide_Cet is -105 ymaxTL: 443, yminTL 151 yminRTL 89 ymaxRTL: 295, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 147


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:  
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 120), (1640, 440), color, 12)
                    logger.debug("HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"Exception in Left Side check: {e}")
                print("Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 80#74#68#60#55#47#40#20
                    thresholdPixel_RightSide_2 = 130#127#120#110#102#90#70
                    HlaSize = ymaxTR -yminTR
                    ymin_R_Value = yminTR -yminRTR # yminTR 640 
                    ymax_R_Value = ymaxTR -ymaxRTR #yminRTR =700
                    RightSide_Cet = cy -Rcy # ok -16
                    logger.debug(f"Right I3, LeftSide_Cet is {RightSide_Cet} yminTR: {yminTR}, ymaxTR {ymaxTR} yminRTR {yminRTR} ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right I3, LeftSide_Cet is -58 yminTR: 781, ymaxTR 1072 yminRTR 882 ymaxRTR: 1086, thresholdPixel_RightSide_1: 47, thresholdPixel_RightSide_2: 90
                    #Right I3, LeftSide_Cet is -55 yminTR: 802, ymaxTR 1094 yminRTR 899 ymaxRTR: 1107, thresholdPixel_RightSide_1: 47, thresholdPixel_RightSide_2: 90
                    #Right I3, LeftSide_Cet is 16 yminTR: 686, ymaxTR 976 yminRTR 713 ymaxRTR: 918, thresholdPixel_RightSide_1: 47, thresholdPixel_RightSide_2: 110
                    #Right I3, LeftSide_Cet is 15 yminTR: 777, ymaxTR 1065 yminRTR 804 ymaxRTR: 1009, thresholdPixel_RightSide_1: 55, thresholdPixel_RightSide_2: 110
                    #Right I3, LeftSide_Cet is 17 yminTR: 764, ymaxTR 1059 yminRTR 792 ymaxRTR: 997, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 110
                    #Right I3, LeftSide_Cet is -64 yminTR: 756, ymaxTR 1058 yminRTR 868 ymaxRTR: 1074, thresholdPixel_RightSide_1: 68, thresholdPixel_RightSide_2: 110
                    #Right I3, LeftSide_Cet is 24 yminTR: 806, ymaxTR 1102 yminRTR 828 ymaxRTR: 1033, thresholdPixel_RightSide_1: 68, thresholdPixel_RightSide_2: 120
                    #Right I3, LeftSide_Cet is -74 yminTR: 757, ymaxTR 1059 yminRTR 878 ymaxRTR: 1087, thresholdPixel_RightSide_1: 74, thresholdPixel_RightSide_2: 120
                    #Right I3, LeftSide_Cet is 29 yminTR: 734, ymaxTR 1037 yminRTR 751 ymaxRTR: 962, thresholdPixel_RightSide_1: 74, thresholdPixel_RightSide_2: 120
                    #Right I3, LeftSide_Cet is -82 yminTR: 800, ymaxTR 1097 yminRTR 928 ymaxRTR: 1132, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 127

                  
                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_1): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1360, 730), (1650, 1050), color, 12)
                    logger.debug("right i3 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I3 Exception in Right Side check: {e}")
                print("I3 Exception in Right Side check:", e)

        #==========================I4====================================
        if Position == "3":  # I4 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500:
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    thresholdPixel_LeftSide_1 = 60#54#43#38#31#26#15
                    thresholdPixel_LeftSide_2 = 153#145#135#115#90#85
                    HlaSize = ymaxTL -yminTL
                    ymin_L_Value = yminTL -yminRTL
                    ymax_L_Value = ymaxTL -ymaxRTL
                    LeftSide_Cet = Lcy - cy 

                   
                    logger.debug(f"Left I4, LeftSide_Cet is {LeftSide_Cet} ymaxTL: {ymaxTL}, yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    

                    #Left I4, LeftSide_Cet is -15 ymaxTL: 507, yminTL 219 yminRTL 243 ymaxRTL: 454, thresholdPixel_LeftSide_1: 15, thresholdPixel_LeftSide_2: 85
                    #Left I4, LeftSide_Cet is -52 ymaxTL: 412, yminTL 133 yminRTL 116 ymaxRTL: 325, thresholdPixel_LeftSide_1: 26, thresholdPixel_LeftSide_2: 85
                    #Left I4, LeftSide_Cet is -10 ymaxTL: 549, yminTL 259 yminRTL 289 ymaxRTL: 500, thresholdPixel_LeftSide_1: 26, thresholdPixel_LeftSide_2: 90
                    #Left I4, LeftSide_Cet is -10 ymaxTL: 549, yminTL 259 yminRTL 289 ymaxRTL: 500, thresholdPixel_LeftSide_1: 26, thresholdPixel_LeftSide_2: 90
                    #Left I4, LeftSide_Cet is -84 ymaxTL: 517, yminTL 228 yminRTL 183 ymaxRTL: 393, thresholdPixel_LeftSide_1: 38, thresholdPixel_LeftSide_2: 90
                    #Left I4, LeftSide_Cet is -84 ymaxTL: 517, yminTL 228 yminRTL 183 ymaxRTL: 393, thresholdPixel_LeftSide_1: 38, thresholdPixel_LeftSide_2: 90
                    #Left I4, LeftSide_Cet is -72 ymaxTL: 535, yminTL 250 yminRTL 214 ymaxRTL: 426, thresholdPixel_LeftSide_1: 38, thresholdPixel_LeftSide_2: 90
                    #Left I4, LeftSide_Cet is -85 ymaxTL: 425, yminTL 134 yminRTL 90 ymaxRTL: 298, thresholdPixel_LeftSide_1: 38, thresholdPixel_LeftSide_2: 115
                    #Left I4, LeftSide_Cet is -81 ymaxTL: 509, yminTL 216 yminRTL 177 ymaxRTL: 386, thresholdPixel_LeftSide_1: 38, thresholdPixel_LeftSide_2: 115
                    #Left I4, LeftSide_Cet is -85 ymaxTL: 491, yminTL 197 yminRTL 153 ymaxRTL: 366, thresholdPixel_LeftSide_1: 43, thresholdPixel_LeftSide_2: 120
                    #Left I4, LeftSide_Cet is -79 ymaxTL: 579, yminTL 285 yminRTL 249 ymaxRTL: 457, thresholdPixel_LeftSide_1: 43, thresholdPixel_LeftSide_2: 120
                    #Left I4, LeftSide_Cet is -85 ymaxTL: 491, yminTL 197 yminRTL 153 ymaxRTL: 366, thresholdPixel_LeftSide_1: 43, thresholdPixel_LeftSide_2: 120
                    #Left I4, LeftSide_Cet is -89 ymaxTL: 574, yminTL 280 yminRTL 234 ymaxRTL: 442, thresholdPixel_LeftSide_1: 43, thresholdPixel_LeftSide_2: 130
                    #Left I4, LeftSide_Cet is -94 ymaxTL: 581, yminTL 285 yminRTL 235 ymaxRTL: 443, thresholdPixel_LeftSide_1: 43, thresholdPixel_LeftSide_2: 135
                    #Left I4, LeftSide_Cet is 14 ymaxTL: 515, yminTL 219 yminRTL 274 ymaxRTL: 489, thresholdPixel_LeftSide_1: 54, thresholdPixel_LeftSide_2: 135
                    #Left I4, LeftSide_Cet is -97 ymaxTL: 477, yminTL 185 yminRTL 128 ymaxRTL: 340, thresholdPixel_LeftSide_1: 54, thresholdPixel_LeftSide_2: 135
                    #Left I4, LeftSide_Cet is -105 ymaxTL: 547, yminTL 251 yminRTL 192 ymaxRTL: 396, thresholdPixel_LeftSide_1: 60, thresholdPixel_LeftSide_2: 145


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                       
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 160), (1640, 510), color, 12)
                    logger.debug("I4 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"I4 Exception in Left Side check: {e}")
                print("I4 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 72#60#50#40#30
                    thresholdPixel_RightSide_2 = 131#121#117#110#105
                    HlaSize = ymaxTR -yminTR
                    ymin_R_Value = yminTR -yminRTR
                    ymax_R_Value = ymaxTR -ymaxRTR
                    RightSide_Cet = cy -Rcy
                    
                  
                    logger.debug(f"Right I4,ymax_R_Value is {ymax_R_Value} RightSide_Cet is {RightSide_Cet} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR},"
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    
                    #Right I4,ymax_R_Value is -31 RightSide_Cet is -71 yminTR is 862 yminRTR 972, ymaxTR: 1157, ymaxRTR: 1188,thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 105
                    #Right I4,ymax_R_Value is -30 RightSide_Cet is -72 yminTR is 812 yminRTR 925, ymaxTR: 1109, ymaxRTR: 1139,thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 110
                    # Right I4,ymax_R_Value is 35 RightSide_Cet is -5 yminTR is 747 yminRTR 793, ymaxTR: 1041, ymaxRTR: 1006,thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 117
                    #Right I4,ymax_R_Value is 32 RightSide_Cet is -7 yminTR is 720 yminRTR 767, ymaxTR: 1012, ymaxRTR: 980,thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 117
                    #Right I4,ymax_R_Value is 44 RightSide_Cet is 1 yminTR is 814 yminRTR 855, ymaxTR: 1109, ymaxRTR: 1065,thresholdPixel_RightSide_1: 40, thresholdPixel_RightSide_2: 117
                    #Right I4,ymax_R_Value is 58 RightSide_Cet is 12 yminTR is 848 yminRTR 882, ymaxTR: 1147, ymaxRTR: 1089,thresholdPixel_RightSide_1: 50, thresholdPixel_RightSide_2: 117
                    #Right I4,ymax_R_Value is 68 RightSide_Cet is 20 yminTR is 813 yminRTR 841, ymaxTR: 1114, ymaxRTR: 1046,thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 117
                    #Right I4,ymax_R_Value is -23 RightSide_Cet is -71 yminTR is 870 yminRTR 989, ymaxTR: 1170, ymaxRTR: 1193,thresholdPixel_RightSide_1: 72, thresholdPixel_RightSide_2: 117
                    #Right I4,ymax_R_Value is -1 RightSide_Cet is -64 yminTR is 897 yminRTR 1024, ymaxTR: 1195, ymaxRTR: 1196,thresholdPixel_RightSide_1: 72, thresholdPixel_RightSide_2: 121

                    
                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_1): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                   
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1360, 770), (1660, 1110), color, 12)
                    logger.debug("I4 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I4 Exception in Right Side check: {e}")
                print("I4 Exception in Right Side check:", e)  

        #======================================= I5
        if Position == "4":  # I5 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500:
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    thresholdPixel_LeftSide_1 = 35#28#20#1
                    thresholdPixel_LeftSide_2 = 147#135#120#110
                    HlaSize = ymaxTL -yminTL
                    ymin_L_Value = yminTL -yminRTL
                    ymax_L_Value = ymaxTL -ymaxRTL
                    LeftSide_Cet = Lcy - cy #ok -6

                    logger.debug(f"Left I5, LeftSide_Cet is {LeftSide_Cet} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    
                    #Left I5, LeftSide_Cet is -19 ymaxTL: 512,yminTL 220 yminRTL236 ymaxRTL: 458, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 90
                    #Left I5, LeftSide_Cet is -62 ymaxTL: 465,yminTL 175 yminRTL150 ymaxRTL: 366, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 90
                    #Left I5, LeftSide_Cet is -68 ymaxTL: 385,yminTL 92 yminRTL52 ymaxRTL: 288, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 90
                    #Left I5, LeftSide_Cet is -60 ymaxTL: 420,yminTL 129 yminRTL99 ymaxRTL: 329, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 90
                    #Left I5, LeftSide_Cet is -85 ymaxTL: 509,yminTL 222 yminRTL 168 ymaxRTL: 393, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 110
                    #Left I5, LeftSide_Cet is -99 ymaxTL: 552,yminTL 260 yminRTL 194 ymaxRTL: 420, thresholdPixel_LeftSide_1: 28, thresholdPixel_LeftSide_2: 120
                    #Left I5, LeftSide_Cet is -1 ymaxTL: 555,yminTL 262 yminRTL 294 ymaxRTL: 520, thresholdPixel_LeftSide_1: 28, thresholdPixel_LeftSide_2: 135
                    #Left I5, LeftSide_Cet is -109 ymaxTL: 548,yminTL 257 yminRTL 184 ymaxRTL: 402, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 135
                    #Left I5, LeftSide_Cet is -109 ymaxTL: 548,yminTL 257 yminRTL 184 ymaxRTL: 402, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 135
                    #Left I5, LeftSide_Cet is -106 ymaxTL: 548,yminTL 255 yminRTL 189 ymaxRTL: 402, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 135
                    #Left I5, LeftSide_Cet is -103 ymaxTL: 504,yminTL 208 yminRTL 143 ymaxRTL: 363, thresholdPixel_LeftSide_1: 35, thresholdPixel_LeftSide_2: 135


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1370, 180), (1650, 520), color, 12)
                    logger.debug("I5 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"I5 Exception in Left Side check: {e}")
                print("I5 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 140#128#122#117#110#80
                    thresholdPixel_RightSide_2 = 62#50#40#30
                    HlaSize = ymaxTR -yminTR
                    ymin_R_Value = yminTR -yminRTR
                    ymax_R_Value = ymaxTR -ymaxRTR
                    RightSide_Cet = cy -Rcy
                    logger.debug(f"Right I5,ymax_R_Value is {ymax_R_Value} RightSide_Cet is {RightSide_Cet} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    # Right I5,ymax_R_Value is -35 RightSide_Cet is -74 yminTR is 773 yminRTR 886, ymaxTR: 1068, ymaxRTR: 1103, thresholdPixel_RightSide_1: 110, thresholdPixel_RightSide_2: 30
                    #Right I5,ymax_R_Value is -35 RightSide_Cet is -74 yminTR is 773 yminRTR 886, ymaxTR: 1068, ymaxRTR: 1103, thresholdPixel_RightSide_1: 110, thresholdPixel_RightSide_2: 30
                    #Right I5,ymax_R_Value is 33 RightSide_Cet is -2 yminTR is 809 yminRTR 846, ymaxTR: 1094, ymaxRTR: 1061, thresholdPixel_RightSide_1: 117, thresholdPixel_RightSide_2: 30
                    #Right I5,ymax_R_Value is 46 RightSide_Cet is 2 yminTR is 848 yminRTR 890, ymaxTR: 1141, ymaxRTR: 1095, thresholdPixel_RightSide_1: 128, thresholdPixel_RightSide_2: 40
                    #Right I5,ymax_R_Value is 57 RightSide_Cet is 19 yminTR is 759 yminRTR 778, ymaxTR: 1050, ymaxRTR: 993, thresholdPixel_RightSide_1: 128, thresholdPixel_RightSide_2: 50
                    #Right I5,ymax_R_Value is -52 RightSide_Cet is -94 yminTR is 822 yminRTR 958, ymaxTR: 1126, ymaxRTR: 1178, thresholdPixel_RightSide_1: 128, thresholdPixel_RightSide_2: 62


                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_2): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                     
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1400, 770), (1670, 1110), color, 12)
                    logger.debug("I5 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I5 Exception in Right Side check: {e}")
                print("I5 Exception in Right Side check:", e)  
                
        
        #======================================= E4
        if Position == "5":  # E4 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500:  # Left-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        if ymax < 180:
                            continue
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]
                    thresholdPixel_LeftSide_1 =  58#55#45#37#27#22#10#5
                    thresholdPixel_LeftSide_2 = 154#140#134# 120#105#97
                    HlaSize = ymaxTL -yminTL
                    ymin_L_Value = yminTL -yminRTL
                    ymax_L_Value = ymaxTL -ymaxRTL
                    LeftSide_Cet = Lcy - cy #not ok 31

                    logger.debug(f"Left E4, LeftSide_Cet is {LeftSide_Cet} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL},, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left E4, LeftSide_Cet is -63 ymaxTL: 466,yminTL 192 yminRTL 165 ymaxRTL: 367,, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 97
                    #Left E4, LeftSide_Cet is -79 ymaxTL: 512,yminTL 234 yminRTL 191 ymaxRTL: 397,, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 105
                    #Left E4, LeftSide_Cet is -35 ymaxTL: 517,yminTL 231 yminRTL 238 ymaxRTL: 441,, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 120
                    #Left E4, LeftSide_Cet is -24 ymaxTL: 556,yminTL 268 yminRTL 286 ymaxRTL: 491,, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 120
                    #Left E4, LeftSide_Cet is -17 ymaxTL: 562,yminTL 276 yminRTL 300 ymaxRTL: 505,, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 120
                    #Left E4, LeftSide_Cet is -17 ymaxTL: 562,yminTL 276 yminRTL 300 ymaxRTL: 505,, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 120
                    #Left E4, LeftSide_Cet is -87 ymaxTL: 519,yminTL 232 yminRTL 187 ymaxRTL: 390,, thresholdPixel_LeftSide_1: 27, thresholdPixel_LeftSide_2: 120
                    ##Left E4, LeftSide_Cet is -10 ymaxTL: 500,yminTL 211 yminRTL 242 ymaxRTL: 448,, thresholdPixel_LeftSide_1: 27, thresholdPixel_LeftSide_2: 134
                    #Left E4, LeftSide_Cet is -10 ymaxTL: 500,yminTL 211 yminRTL 242 ymaxRTL: 448,, thresholdPixel_LeftSide_1: 27, thresholdPixel_LeftSide_2: 134
                    #Left E4, LeftSide_Cet is -12 ymaxTL: 601,yminTL 303 yminRTL 341 ymaxRTL: 540,, thresholdPixel_LeftSide_1: 37, thresholdPixel_LeftSide_2: 134
                    #Left E4, LeftSide_Cet is 12 ymaxTL: 573,yminTL 282 yminRTL 338 ymaxRTL: 540,, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 134
                    #Left E4, LeftSide_Cet is 12 ymaxTL: 573,yminTL 282 yminRTL 338 ymaxRTL: 540,, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 134
                    #Left E4, LeftSide_Cet is -184 ymaxTL: 582,yminTL 287 yminRTL 180 ymaxRTL: 320,, thresholdPixel_LeftSide_1: 55, thresholdPixel_LeftSide_2: 134
                    #Left E4, LeftSide_Cet is -91 ymaxTL: 502,yminTL 209 yminRTL 162 ymaxRTL: 367,, thresholdPixel_LeftSide_1: 55, thresholdPixel_LeftSide_2: 134
                    #Left E4, LeftSide_Cet is 10 ymaxTL: 552,yminTL 262 yminRTL 318 ymaxRTL: 517,, thresholdPixel_LeftSide_1: 55, thresholdPixel_LeftSide_2: 140
                    #Left E4, LeftSide_Cet is -109 ymaxTL: 470,yminTL 184 yminRTL 117 ymaxRTL: 319,, thresholdPixel_LeftSide_1: 58, thresholdPixel_LeftSide_2: 140


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = " "
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (500, 170), (775, 510), color, 12)
                    logger.debug("E4 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E4 Exception in Left Side check: {e}")
                print("E4 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 110#102#94#86#82#75#65
                    thresholdPixel_RightSide_2 = 77#75#70#50
                    HlaSize = ymaxTR - yminTR
                    ymin_R_Value = yminTR -yminRTR
                    ymax_R_Value = ymaxTR -ymaxRTR
                    RightSide_Cet = cy -Rcy  #ok 23

                    logger.debug(f"Right E4,RightSide_Cet is {RightSide_Cet} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    
                    #Right E4,RightSide_Cet is -10 yminTR is 699 yminRTR 746, ymaxTR: 982, ymaxRTR: 954, thresholdPixel_RightSide_1: 75, thresholdPixel_RightSide_2: 50
                    #Right E4,RightSide_Cet is 21 yminTR is 757 yminRTR 778, ymaxTR: 1043, ymaxRTR: 980, thresholdPixel_RightSide_1: 75, thresholdPixel_RightSide_2: 50
                    #Right E4,RightSide_Cet is -40 yminTR is 760 yminRTR 838, ymaxTR: 1041, ymaxRTR: 1042, thresholdPixel_RightSide_1: 75, thresholdPixel_RightSide_2: 70
                    #Right E4,RightSide_Cet is -48 yminTR is 733 yminRTR 816, ymaxTR: 1017, ymaxRTR: 1031, thresholdPixel_RightSide_1: 82, thresholdPixel_RightSide_2: 70
                    #Right E4,RightSide_Cet is -59 yminTR is 813 yminRTR 911, ymaxTR: 1098, ymaxRTR: 1118, thresholdPixel_RightSide_1: 94, thresholdPixel_RightSide_2: 70
                    #Right E4,RightSide_Cet is -63 yminTR is 818 yminRTR 921, ymaxTR: 1106, ymaxRTR: 1129, thresholdPixel_RightSide_1: 102, thresholdPixel_RightSide_2: 70
                    #Right E4,RightSide_Cet is 27 yminTR is 742 yminRTR 761, ymaxTR: 1034, ymaxRTR: 962, thresholdPixel_RightSide_1: 110, thresholdPixel_RightSide_2: 70
                    #Right E4,RightSide_Cet is 31 yminTR is 692 yminRTR 704, ymaxTR: 977, ymaxRTR: 902, thresholdPixel_RightSide_1: 110, thresholdPixel_RightSide_2: 70

                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_2): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (500, 700), (775, 1040), color, 12)
                    logger.debug("E4 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"E4 Exception in Right Side check: {e}")
                print("E4 Exception in Right Side check:", e)  

        #======================================= E3
        if Position == "6":  # E3 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        if smalllabellist[1] < 950:
                            hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]
                    thresholdPixel_LeftSide_1 = 50#32#21#9#5#2
                    thresholdPixel_LeftSide_2 = 145#125#115#105
                    HlaSize = ymaxTL - yminTL
                    ymin_L_Value = yminTL -yminRTL
                    ymax_L_Value = ymaxTL -ymaxRTL
                    LeftSide_Cet = Lcy - cy #not ok 40

                    logger.debug(f"Left E3,LeftSide_Cet is {LeftSide_Cet} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left E3,LeftSide_Cet is -70 ymaxTL: 465,yminTL 181 yminRTL 151 ymaxRTL: 356, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 105
                    #Left E3,LeftSide_Cet is -33 ymaxTL: 487,yminTL 200 yminRTL 206 ymaxRTL: 415, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 115
                    #Left E3,LeftSide_Cet is -21 ymaxTL: 520,yminTL 239 yminRTL 255 ymaxRTL: 461, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 115
                    #Left E3,LeftSide_Cet is -35 ymaxTL: 483,yminTL 194 yminRTL 201 ymaxRTL: 406, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 115
                    #Left E3,LeftSide_Cet is -35 ymaxTL: 483,yminTL 194 yminRTL 201 ymaxRTL: 406, thresholdPixel_LeftSide_1: 5, thresholdPixel_LeftSide_2: 115

                    #Left E3,LeftSide_Cet is -80 ymaxTL: 439,yminTL 153 yminRTL 111 ymaxRTL: 321, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 115
                    #Left E3,LeftSide_Cet is -93 ymaxTL: 436,yminTL 146 yminRTL 91 ymaxRTL: 305, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 125
                    #Left E2,LeftSide_Cet is -11 ymaxTL: 493,yminTL 205 yminRTL 234 ymaxRTL: 442, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 151
                    #Left E3,LeftSide_Cet is -10 ymaxTL: 507,yminTL 225 yminRTL 254 ymaxRTL: 459, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 135
                    # Left E3,LeftSide_Cet is -96 ymaxTL: 428,yminTL 133 yminRTL 78 ymaxRTL: 291, thresholdPixel_LeftSide_1: 32, thresholdPixel_LeftSide_2: 135
                    #Left E3,LeftSide_Cet is -96 ymaxTL: 428,yminTL 133 yminRTL 78 ymaxRTL: 291, thresholdPixel_LeftSide_1: 32, thresholdPixel_LeftSide_2: 135
                    #Left E3,LeftSide_Cet is -33 ymaxTL: 519,yminTL 224 yminRTL 234 ymaxRTL: 442, thresholdPixel_LeftSide_1: 32, thresholdPixel_LeftSide_2: 135
                    #Left E3,LeftSide_Cet is 0 ymaxTL: 518,yminTL 223 yminRTL 266 ymaxRTL: 475, thresholdPixel_LeftSide_1: 32, thresholdPixel_LeftSide_2: 145


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2):   #261 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (485, 170), (770, 490), color, 12)
                    logger.debug("E3 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E3 Exception in Left Side check: {e}")
                print("E3 Exception in Left Side check:", e)

            # Right Side check
            try:  
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 112#106#95#82#81#66#60
                    thresholdPixel_RightSide_2 = 82#71#65#60#35#25#15#65
                    HlaSize = ymaxTR - yminTR
                    ymin_R_Value = yminTR -yminRTR
                    ymax_R_Value = ymaxTR -ymaxRTR
                    
                    RightSide_Cet = cy -Rcy
                    logger.debug(f"Right E3, RightSide_Cet is {RightSide_Cet} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR},"
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right E3, RightSide_Cet is -25 yminTR is 742 yminRTR 803, ymaxTR: 1021, ymaxRTR: 1010,thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 65
                    #Right E3, RightSide_Cet is -36 yminTR is 747 yminRTR 821, ymaxTR: 1029, ymaxRTR: 1028,thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 65
                    #Right E3, RightSide_Cet is -24 yminTR is 728 yminRTR 790, ymaxTR: 1008, ymaxRTR: 995,thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 65
                    #Right E3, RightSide_Cet is 32 yminTR is 711 yminRTR 722, ymaxTR: 1003, ymaxRTR: 928,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 65
                    #Right E3, RightSide_Cet is 26 yminTR is 741 yminRTR 756, ymaxTR: 1028, ymaxRTR: 961,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 15
                    #Right E3, RightSide_Cet is -15 yminTR is 720 yminRTR 771, ymaxTR: 998, ymaxRTR: 978,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 15
                    #Right E3, RightSide_Cet is -10 yminTR is 743 yminRTR 791, ymaxTR: 1028, ymaxRTR: 999,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 15
                    #Right E3, RightSide_Cet is -16 yminTR is 741 yminRTR 794, ymaxTR: 1023, ymaxRTR: 1002,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 15
                    #Right E3, RightSide_Cet is -12 yminTR is 721 yminRTR 771, ymaxTR: 1006, ymaxRTR: 979,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 15
                    #Right E3, RightSide_Cet is -12 yminTR is 721 yminRTR 771, ymaxTR: 1006, ymaxRTR: 979,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 15
                    #Right E3, RightSide_Cet is 8 yminTR is 733 yminRTR 763, ymaxTR: 1015, ymaxRTR: 970,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 35
                    #Right E3, RightSide_Cet is 2 yminTR is 700 yminRTR 737, ymaxTR: 978, ymaxRTR: 938,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 35
                    #Right E3, RightSide_Cet is 24 yminTR is 603 yminRTR 616, ymaxTR: 884, ymaxRTR: 822,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 60
                    # Right E3, RightSide_Cet is 30 yminTR is 667 yminRTR 674, ymaxTR: 947, ymaxRTR: 881,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 65
                    #Right E3, RightSide_Cet is -47 yminTR is 714 yminRTR 796, ymaxTR: 999, ymaxRTR: 1011,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 71
                    #Right E3, RightSide_Cet is -54 yminTR is 744 yminRTR 835, ymaxTR: 1024, ymaxRTR: 1042,thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 71
                    #Right E3, RightSide_Cet is -64 yminTR is 753 yminRTR 855, ymaxTR: 1034, ymaxRTR: 1060,thresholdPixel_RightSide_1: 95, thresholdPixel_RightSide_2: 71
                    #Right E3, RightSide_Cet is -66 yminTR is 774 yminRTR 881, ymaxTR: 1063, ymaxRTR: 1088,thresholdPixel_RightSide_1: 106, thresholdPixel_RightSide_2: 71
                    #Right E3, RightSide_Cet is 37 yminTR is 689 yminRTR 694, ymaxTR: 976, ymaxRTR: 896,thresholdPixel_RightSide_1: 112, thresholdPixel_RightSide_2: 71


                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_2): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (490, 685), (770, 1010), color, 12)
                    logger.debug("E3 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"E3 Exception in Right Side check: {e}")
                print("E3 Exception in Right Side check:", e)  

        #======================================= E2
        if Position == "7":  # E2 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 430: #500
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        if smalllabellist[1] < 950:
                            hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]
                    HlaSize = ymaxTL - yminTL
                    thresholdPixel_LeftSide_1 = 45#30#21#15#6#4#1#70
                    thresholdPixel_LeftSide_2 = 151#127#121

                    ymin_L_Value = yminTL -yminRTL
                    ymax_L_Value = ymaxTL -ymaxRTL
                    LeftSide_Cet = Lcy - cy # not ok 7  not 27

                    logger.debug(f"Left E2,LeftSide_Cet is {LeftSide_Cet} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left E2,LeftSide_Cet is -37 ymaxTL: 481,yminTL 196 yminRTL 198 ymaxRTL: 405, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 115
                    #Left E2,LeftSide_Cet is -79 ymaxTL: 489,yminTL 207 yminRTL 165 ymaxRTL: 373, thresholdPixel_LeftSide_1: 6, thresholdPixel_LeftSide_2: 115
                    #Left E2,LeftSide_Cet is -29 ymaxTL: 522,yminTL 234 yminRTL 245 ymaxRTL: 454, thresholdPixel_LeftSide_1: 6, thresholdPixel_LeftSide_2: 121
                    #Left E2,LeftSide_Cet is -24 ymaxTL: 504,yminTL 218 yminRTL 234 ymaxRTL: 440, thresholdPixel_LeftSide_1: 15, thresholdPixel_LeftSide_2: 121
                    #Left E2,LeftSide_Cet is -84 ymaxTL: 468,yminTL 183 yminRTL 136 ymaxRTL: 346, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 121
                    #Left E2,LeftSide_Cet is -95 ymaxTL: 449,yminTL 158 yminRTL 104 ymaxRTL: 313, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 127
                    #Left E2,LeftSide_Cet is -10 ymaxTL: 547,yminTL 253 yminRTL 288 ymaxRTL: 492, thresholdPixel_LeftSide_1: 30, thresholdPixel_LeftSide_2: 151


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                   
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (480, 200), (75, 510), color, 12)
                    logger.debug("E2 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E2 Exception in Left Side check: {e}")
                print("E2 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 86#80#70#60
                    thresholdPixel_RightSide_2 = 107#100#84#80#70#61#57#45
                    HlaSize = ymaxTR - yminTR
                    ymin_R_Value = yminTR -yminRTR
                    ymax_R_Value = ymaxTR -ymaxRTR
                    RightSide_Cet = cy -Rcy

                    logger.debug(f"Right E2,RightSide_Cet is {RightSide_Cet} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right E2,RightSide_Cet is -16 yminTR is 722 yminRTR 780, ymaxTR: 1012, ymaxRTR: 987, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 57

                    #Right E2,RightSide_Cet is -16 yminTR is 764 yminRTR 822, ymaxTR: 1053, ymaxRTR: 1026, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 57
                    #Right E2,RightSide_Cet is -27 yminTR is 747 yminRTR 814, ymaxTR: 1033, ymaxRTR: 1021, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 61
                    #Right E2,RightSide_Cet is -32 yminTR is 753 yminRTR 827, ymaxTR: 1047, ymaxRTR: 1037, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 70
                    #Right E2,RightSide_Cet is 23 yminTR is 654 yminRTR 672, ymaxTR: 940, ymaxRTR: 876, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 80
                    #Right E2,RightSide_Cet is -58 yminTR is 768 yminRTR 867, ymaxTR: 1059, ymaxRTR: 1075, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 84
                    #Right E2,RightSide_Cet is -52 yminTR is 776 yminRTR 868, ymaxTR: 1063, ymaxRTR: 1075, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 84
                    #Right E2,RightSide_Cet is -54 yminTR is 847 yminRTR 939, ymaxTR: 1134, ymaxRTR: 1149, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 84
                    #Right E2,RightSide_Cet is -62 yminTR is 823 yminRTR 924, ymaxTR: 1114, ymaxRTR: 1136, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 100
                    #Right E2,RightSide_Cet is -62 yminTR is 823 yminRTR 924, ymaxTR: 1114, ymaxRTR: 1136, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 100
                    #Right E2,RightSide_Cet is 41 yminTR is 704 yminRTR 707, ymaxTR: 992, ymaxRTR: 908, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 107


                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_1): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (490, 715), (740, 1020), color, 12)
                    logger.debug("E2 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"E2 Exception in Right Side check: {e}")
                print("E2 Exception in Right Side check:", e)  

        #======================================= E1
        if Position == "8":  # E1 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])
 
                if smalllabellist[2] < 580: #500
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                      #  hla_label_dict_right['0'] = smalllabellist
                  #  elif smalllabellist[0] == 'HLA_ROD_TOP':
                    elif smalllabellist[0] == 'HLA_ROD_TOP' and ymax < 1150:
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    
                    thresholdPixel_LeftSide_1 = 40#27#22#15#10#7#1
                    thresholdPixel_LeftSide_2 = 144#140#135#130#120#112
                    HlaSize = ymaxTL - yminTL
                    ymin_L_Value = yminTL -yminRTL
                    ymax_L_Value = ymaxTL -ymaxRTL
                    LeftSide_Cet = Lcy - cy # not ok 25

                    logger.debug(f"Left E1, LeftSide_Cet is {LeftSide_Cet} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left E1, LeftSide_Cet is -68 ymaxTL: 433,yminTL 135 yminRTL 106 ymaxRTL: 326, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 105
                    #Left E1, LeftSide_Cet is -37 ymaxTL: 527,yminTL 230 yminRTL 233 ymaxRTL: 450, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 112
                    #Left E1, LeftSide_Cet is -30 ymaxTL: 505,yminTL 210 yminRTL 217 ymaxRTL: 437, thresholdPixel_LeftSide_1: 1, thresholdPixel_LeftSide_2: 112
                    #Left E1, LeftSide_Cet is -25 ymaxTL: 555,yminTL 257 yminRTL 268 ymaxRTL: 494, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 112
                    # Left E1, LeftSide_Cet is -81 ymaxTL: 439,yminTL 152 yminRTL 104 ymaxRTL: 324, thresholdPixel_LeftSide_1: 15, thresholdPixel_LeftSide_2: 112
                    #Left E1, LeftSide_Cet is -91 ymaxTL: 429,yminTL 136 yminRTL 80 ymaxRTL: 302, thresholdPixel_LeftSide_1: 15, thresholdPixel_LeftSide_2: 120
                    #Left E1, LeftSide_Cet is -99 ymaxTL: 440,yminTL 142 yminRTL 79 ymaxRTL: 305, thresholdPixel_LeftSide_1: 15, thresholdPixel_LeftSide_2: 120
                    #Left E1, LeftSide_Cet is -19 ymaxTL: 450,yminTL 154 yminRTL 171 ymaxRTL: 395, thresholdPixel_LeftSide_1: 15, thresholdPixel_LeftSide_2: 120
                    #Left E1, LeftSide_Cet is -15 ymaxTL: 497,yminTL 199 yminRTL 222 ymaxRTL: 445, thresholdPixel_LeftSide_1: 22, thresholdPixel_LeftSide_2: 130
                    #Left E1, LeftSide_Cet is -4 ymaxTL: 529,yminTL 233 yminRTL 266 ymaxRTL: 488, thresholdPixel_LeftSide_1: 27, thresholdPixel_LeftSide_2: 130
                    #Left E1, LeftSide_Cet is -92 ymaxTL: 458,yminTL 153 yminRTL 101 ymaxRTL: 325, thresholdPixel_LeftSide_1: 40, thresholdPixel_LeftSide_2: 130
                    #Left E1, LeftSide_Cet is -100 ymaxTL: 528,yminTL 231 yminRTL 168 ymaxRTL: 390, thresholdPixel_LeftSide_1: 40, thresholdPixel_LeftSide_2: 135
                    #Left E1, LeftSide_Cet is -101 ymaxTL: 470,yminTL 172 yminRTL 112 ymaxRTL: 329, thresholdPixel_LeftSide_1: 40, thresholdPixel_LeftSide_2: 140

                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (520, 230), (770, 560), color, 12)
                    logger.debug("E1 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E1 Exception in Left Side check: {e}")
                print("E1 Exception in Left Side check:", e)

            # Right Side check
            try:              
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 90#80
                    thresholdPixel_RightSide_2 = 66#63 #52#45#40#30
                    HlaSize = ymaxTR - yminTR
                    ymin_R_Value = yminTR -yminRTR
                    ymax_R_Value = ymaxTR -ymaxRTR
                    RightSide_Cet = cy -Rcy

                    logger.debug(f"Right E1, RightSide_Cet is {RightSide_Cet} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right E1, RightSide_Cet is 1 yminTR is 706 yminRTR 738, ymaxTR: 994, ymaxRTR: 961, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 30
                    #Right E1, RightSide_Cet is -11 yminTR is 716 yminRTR 757, ymaxTR: 1011, ymaxRTR: 992, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 40
                    #Right E1, RightSide_Cet is 50 yminTR is 665 yminRTR 649, ymaxTR: 961, ymaxRTR: 877, thresholdPixel_RightSide_1: 80, thresholdPixel_RightSide_2: 45
                    #Right E1, RightSide_Cet is -317 yminTR is 632 yminRTR 964, ymaxTR: 894, ymaxRTR: 1196, thresholdPixel_RightSide_1: 90, thresholdPixel_RightSide_2: 45
                    #Right E1, RightSide_Cet is -21 yminTR is 765 yminRTR 823, ymaxTR: 1062, ymaxRTR: 1046, thresholdPixel_RightSide_1: 90, thresholdPixel_RightSide_2: 52
                    # Right E1, RightSide_Cet is -41 yminTR is 782 yminRTR 855, ymaxTR: 1078, ymaxRTR: 1087, thresholdPixel_RightSide_1: 90, thresholdPixel_RightSide_2: 63
                    #Right E1, RightSide_Cet is -32 yminTR is 665 yminRTR 729, ymaxTR: 959, ymaxRTR: 959, thresholdPixel_RightSide_1: 90, thresholdPixel_RightSide_2: 63

                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_1): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (530, 760), (770, 1100), color, 12)
                    logger.debug("E1 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"E1 Exception in Right Side check: {e}")
                print("E1 Exception in Right Side check:", e)  

        #=======================================    status
        if HLA_POSTION_LEFT == "OK" and HLA_POSTION_RIGHT == "OK":  
            HLA_POSTION = "OK"
             
        else:
            HLA_POSTION ="NOT OK"

        
    except Exception as e:
        logger.debug(f"except Exception HLA_PositionCheck_WITH_ROD_4CYLINDER {e}")
        print("HLA_PositionCheck_WITH_ROD_4CYLINDER is :",e)
        HLA_POSTION ="NOT OK"
        return HLA_POSTION, original_image

    return HLA_POSTION, original_image



def HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(Position, OBJECT_LIST, original_image): 
    # Initialize statuses and variables
    HLA_POSTION_RIGHT = "NOT OK"
    HLA_POSTION_LEFT = "NOT OK"
    HLA_POSTION = "NOT OK"

    thresholdPixel_LeftSide_1 = 70
    thresholdPixel_LeftSide_2 = 20

    thresholdPixel_RightSide_1 = 70
    thresholdPixel_RightSide_2 = 40


    try:
        if Position == "1":  # I2 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 530:#560:#630: #500 # Left-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP' and ymax > 100: #Ymin -60 #5
                        hla_label_dict_left['1'] = smalllabellist

                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP' and ymin < 1050: #930
                        hla_label_dict_right['1'] = smalllabellist


            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    thresholdPixel_LeftSide_1 = 30#55
                    thresholdPixel_LeftSide_2 = 30#50#47
                    LeftSide_Cet = Lcy - cy   

                    logger.debug(f"Left I2, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is {LeftSide_Cet}, yminTL {yminTL} ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, yminRTL: {yminRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}") 
                    
                    #Notok =ymaxTL = 427 ymaxRTL = 392 thresholdPixel_LeftSide_2 = "nok"
                    #ok =ymaxTL = 335 ymaxRTL = 337 thresholdPixel_LeftSide_2 = 50
                    #ok =ymaxTL = 397 ymaxRTL = 391 thresholdPixel_LeftSide_2 = 50

                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 160), (1600, 515), color, 12)
                    logger.debug("I2 left HLA_TOP or HLA_ROD_TOP missingc50 on the left side.")
            
            except Exception as e:
                logger.debug(f"I2 HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 Exception in Left Side check: {e}")
                print("I2 HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 Exception in Left Side check:", e)

            # Right Side check
            try:              
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    thresholdPixel_RightSide_1 = 81#78#67 #60
                    thresholdPixel_RightSide_2 = 3#55#57
                   
                    logger.debug(f"Right I2,ymax_R_Value_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")

                    #Right I2,ymax_R_Value_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 727 yminRTR 788, ymaxTR: 985, ymaxRTR: 1039, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 55
                    #Right I2,ymax_R_Value_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 697 yminRTR 772, ymaxTR: 956, ymaxRTR: 1023, thresholdPixel_RightSide_1: 67, thresholdPixel_RightSide_2: 3
                    #Right I2,ymax_R_Value_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 733 yminRTR 801, ymaxTR: 990, ymaxRTR: 1048, thresholdPixel_RightSide_1: 67, thresholdPixel_RightSide_2: 3
                    #Notok = ymaxTR = 1023 ymaxRTR = 1017 thresholdPixel_RightSide_2 = "nok"
                    #ok = ymaxTR = 966 ymaxRTR = 1018 thresholdPixel_RightSide_2 = 55
                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_2) : 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                      
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 760), (1600, 1110), color, 12)
                    logger.debug("I2 right HLA_TOP_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I2 Exception in Right Side_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 check: {e}")
                print("I2 Exception in Right Side HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 check:", e)



        #==========================I3====================================
        if Position == "2":  # I3 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500:
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    thresholdPixel_LeftSide_1 = 22
                    thresholdPixel_LeftSide_2 = 22#25#38 #40

                   
                    logger.debug(f"Left I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is ymaxTL: {ymaxTL}, yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")

                    #ok = = ymaxTL = 432 ymaxRTL = 439 thresholdPixel_LeftSide_2 = 22
                    #notok = ymaxTL = 483 ymaxRTL = 450 thresholdPixel_LeftSide_2 = 22
           
                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:  
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 120), (1640, 440), color, 12)
                    logger.debug("HLA_TOP or HLA_ROD_TOP_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 missing on the left side.")
            
            except Exception as e:
                logger.debug(f"Exception in Left_Sidecheck_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("Exception in Left_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    
                    thresholdPixel_RightSide_1 = 96#87#81#75#57
                    thresholdPixel_RightSide_2 = 24#40#80#15#50

                    logger.debug(f"Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: {yminTR}, ymaxTR {ymaxTR} yminRTR {yminRTR} ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 715, ymaxTR 968 yminRTR 792 ymaxRTR: 1035, thresholdPixel_RightSide_1: 75, thresholdPixel_RightSide_2: 15
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 736, ymaxTR 989 yminRTR 812 ymaxRTR: 1051, thresholdPixel_RightSide_1: 75, thresholdPixel_RightSide_2: 15
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 744, ymaxTR 997 yminRTR 828 ymaxRTR: 1064, thresholdPixel_RightSide_1: 75, thresholdPixel_RightSide_2: 15
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 739, ymaxTR 998 yminRTR 821 ymaxRTR: 1074, thresholdPixel_RightSide_1: 81, thresholdPixel_RightSide_2: 80
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 755, ymaxTR 1013 yminRTR 849 ymaxRTR: 1099, thresholdPixel_RightSide_1: 87, thresholdPixel_RightSide_2: 40
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 727, ymaxTR 982 yminRTR 771 ymaxRTR: 1013, thresholdPixel_RightSide_1: 96, thresholdPixel_RightSide_2: 40
                    #ok    = ymaxTR = 994 ymaxRTR = 1066 thresholdPixel_RightSide_2 = 80
                    #notok = ymaxTR = 1046 ymaxRTR = 1075 thresholdPixel_RightSide_2 = 80
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 602, ymaxTR 859 yminRTR 649 ymaxRTR: 885, thresholdPixel_RightSide_1: 96, thresholdPixel_RightSide_2: 40
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 609, ymaxTR 869 yminRTR 662 ymaxRTR: 903, thresholdPixel_RightSide_1: 96, thresholdPixel_RightSide_2: 40
                    #Right I3, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR: 617, ymaxTR 875 yminRTR 668 ymaxRTR: 905, thresholdPixel_RightSide_1: 96, thresholdPixel_RightSide_2: 40

            
                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR - thresholdPixel_RightSide_2): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1350, 685), (1570, 1040), color, 12)
                    logger.debug(" i3 right HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I3 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("I3 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)

        #==========================I4====================================
        if Position == "3":  # I4 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500:
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    thresholdPixel_LeftSide_1 = 14#30
                    thresholdPixel_LeftSide_2 = 55

                   
                    logger.debug(f"Left I4, LeftSide_Cet_Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is ymaxTL: {ymaxTL}, yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    
                    #ok = = ymaxTL = 432 ymaxRTL = 439 thresholdPixel_LeftSide_2 = 22
                    #notok = ymaxTL = 505 ymaxRTL = 493 thresholdPixel_LeftSide_2 = 55

                    #notok = yminTL = 238 yminRTL = 257 thresholdPixel_LeftSide_1 = 30
                    #ok = yminTL = 251 yminRTL = 253 thresholdPixel_LeftSide_1 = 30
                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                       
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1340, 160), (1640, 510), color, 12)
                    logger.debug("I4 HLA_TOP or HLA_ROD_TOP_Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 missing on the left side.")
            
            except Exception as e:
                logger.debug(f"I4 Exception in Left Side check Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("I4 Exception in Left Side check Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 16#22#25 #0
                    thresholdPixel_RightSide_2 = 80#75#70#80#70#65#60#55#50
                    
                  
                    logger.debug(f"Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR},"
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
           
                    #Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 770 yminRTR 831, ymaxTR: 1028, ymaxRTR: 1070,thresholdPixel_RightSide_1: 0, thresholdPixel_RightSide_2: 60
                    #Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 765 yminRTR 832, ymaxTR: 1021, ymaxRTR: 1074,thresholdPixel_RightSide_1: 0, thresholdPixel_RightSide_2: 65
                    # Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 807 yminRTR 879, ymaxTR: 1059, ymaxRTR: 1118,thresholdPixel_RightSide_1: 0, thresholdPixel_RightSide_2: 70
                   # Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 814 yminRTR 885, ymaxTR: 1061, ymaxRTR: 1130,thresholdPixel_RightSide_1: 25, thresholdPixel_RightSide_2: 70
                    #Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 802 yminRTR 874, ymaxTR: 1059, ymaxRTR: 1115,thresholdPixel_RightSide_1: 25, thresholdPixel_RightSide_2: 70
                    #Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 838 yminRTR 914, ymaxTR: 1083, ymaxRTR: 1148,thresholdPixel_RightSide_1: 25, thresholdPixel_RightSide_2: 75

                    #ok    = ymaxTR = 994 ymaxRTR = 1066 thresholdPixel_RightSide_2 = -
                    #notok = ymaxTR = 1100 ymaxRTR = 1115 thresholdPixel_RightSide_1 = 0

                    #notok = yminTR = 795 yminRTR = 870 thresholdPixel_RightSide_2 = 80
                    #ok = yminTR = 807 yminRTR = 864 thresholdPixel_RightSide_2 = 80
                    #Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 671 yminRTR 717, ymaxTR: 928, ymaxRTR: 952,thresholdPixel_RightSide_1: 25, thresholdPixel_RightSide_2: 80
                    #Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 664 yminRTR 707, ymaxTR: 921, ymaxRTR: 940,thresholdPixel_RightSide_1: 22, thresholdPixel_RightSide_2: 80
                    #Right I4, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 686 yminRTR 727, ymaxTR: 941, ymaxRTR: 961,thresholdPixel_RightSide_1: 22, thresholdPixel_RightSide_2: 80

                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR - thresholdPixel_RightSide_1): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                   
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1035, 750), (1280, 1080), color, 12)
                    logger.debug("I4 right HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I4 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("I4 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)  

        #======================================= I5
        if Position == "4":  # I5 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500:
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    thresholdPixel_LeftSide_1 = 30#20
                    thresholdPixel_LeftSide_2 = 52#44#36#33#28#50#30#12

                    logger.debug(f"Left I5, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is  ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    
                    #Left I5, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is  ymaxTL: 427,yminTL 188 yminRTL 153 ymaxRTL: 397, thresholdPixel_LeftSide_1: 30, thresholdPixel_LeftSide_2: 28
                    #Left I5, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is  ymaxTL: 427,yminTL 188 yminRTL 153 ymaxRTL: 397, thresholdPixel_LeftSide_1: 30, thresholdPixel_LeftSide_2: 28
                    #Left I5, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is  ymaxTL: 437,yminTL 203 yminRTL 151 ymaxRTL: 395, thresholdPixel_LeftSide_1: 30, thresholdPixel_LeftSide_2: 36
                    #Left I5, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is  ymaxTL: 448,yminTL 215 yminRTL 158 ymaxRTL: 398, thresholdPixel_LeftSide_1: 30, thresholdPixel_LeftSide_2: 44

                    #ok = = ymaxTL = 472 ymaxRTL = 466 thresholdPixel_LeftSide_2 = 50
                    #notok = ymaxTL = 504 ymaxRTL = 468 thresholdPixel_LeftSide_2 = 50
          
                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2): 
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1370, 180), (1650, 520), color, 12)
                    logger.debug("I5 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"I5 Exception in Left_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("I5 Exception in Left_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    thresholdPixel_RightSide_1 = 60#50#43#117
                    thresholdPixel_RightSide_2 = 15#20

                    logger.debug(f"Right I5, HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    
                    #Right I5, HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 756 yminRTR 802, ymaxTR: 998, ymaxRTR: 1055, thresholdPixel_RightSide_1: 43, thresholdPixel_RightSide_2: 15
                    #Right I5, HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is yminTR is 757 yminRTR 810, ymaxTR: 1001, ymaxRTR: 1048, thresholdPixel_RightSide_1: 50, thresholdPixel_RightSide_2: 15

                    #notok = ymaxTR = 1083 ymaxRTR = 1093 thresholdPixel_RightSide_2 = 15
                    #OK    = ymaxTR = 1039 ymaxRTR = 1084 thresholdPixel_RightSide_2 = 15

                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR - thresholdPixel_RightSide_2): 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                     
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1400, 770), (1670, 1110), color, 12)
                    logger.debug("I5 right HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I5 Exception in Right Side check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("I5 Exception in Right Side check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)  
                
        
        #======================================= E4
        if Position == "5":  # E4 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}
            hla_label_dict_left_line = {}
            hla_label_dict_right_line = {}


            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]

                if smalllabellist[2] < 550:  # Left-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        if ymax < 180:
                            continue
                        hla_label_dict_left['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_left_line['2'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_right_line['2'] = smalllabellist
            
            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    hla_top_L_line = hla_label_dict_left_line.get('2')

                    if hla_top_L_line is None:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTL_line, yminTL_line, xmaxTL_line, ymaxTL_line,result, cx_line, cy_line = hla_top_L_line[1:8]

                    thresholdPixel_LeftSide_1 = 10
                    thresholdPixel_LeftSide_2 = 82#75#16#57#55 #60
                    SizeOf_cy_line = cy_line -cy
                    Check_Line_distance_with_hla = ymaxTL - ymaxTL_line #ok-80
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL #ok -104

                    logger.debug(f"Left E4, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is {SizeOf_cy_line} Check_Line_distance_with_hla is {Check_Line_distance_with_hla}, Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    
                    #Left E4, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 12 Check_Line_distance_with_hla is 78, Check_Line_distance_with_hla_ymin is 102 ymaxTL: 458,yminTL 201 yminRTL 132 ymaxRTL: 381, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 75
                    #Left E4, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 1 Check_Line_distance_with_hla is 95, Check_Line_distance_with_hla_ymin is 98 ymaxTL: 500,yminTL 240 yminRTL 201 ymaxRTL: 449, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 82


                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or SizeOf_cy_line > 27 or Check_Line_distance_with_hla > 96 or Check_Line_distance_with_hla_ymin < 90:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = " "
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (500, 170), (775, 510), color, 12)
                    logger.debug("E4 HLA_TOP or HLA_ROD_TOP_missing_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 on the left side.")
            
            except Exception as e:
                logger.debug(f"E4 Exception in Left Side check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E4 Exception in Left Side check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    hla_top_R_line = hla_label_dict_right_line.get('2')
                    if hla_top_R_line is None:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    xminTR_line, yminTR_line, xmaxTR_line, ymaxTR_line,result,cx_line, cy_line = hla_top_R_line[1:8]


                    thresholdPixel_RightSide_1 = 45 #30#20
                    thresholdPixel_RightSide_2 = 30#25#21#15#30
                    SizeOf_cy_line = cy_line -cy

                    Check_Line_distance_with_hla = ymaxTR - ymaxTR_line #ok -77
                    Check_Line_distance_with_hla_ymin = yminTR_line - yminTR  #ok -94

                    logger.debug(f"Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is {SizeOf_cy_line}, Check_Line_distance_with_hla is {Check_Line_distance_with_hla},Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")

                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 9, Check_Line_distance_with_hla is 81,Check_Line_distance_with_hla_ymin is 99 yminTR is 841 yminRTR 881, ymaxTR: 1099, ymaxRTR: 1121, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 15
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 11, Check_Line_distance_with_hla is 79,Check_Line_distance_with_hla_ymin is 100 yminTR is 774 yminRTR 769, ymaxTR: 1029, ymaxRTR: 1011, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 15
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 4, Check_Line_distance_with_hla is 90,Check_Line_distance_with_hla_ymin is 97 yminTR is 851 yminRTR 884, ymaxTR: 1110, ymaxRTR: 1125, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 4, Check_Line_distance_with_hla is 90,Check_Line_distance_with_hla_ymin is 97 yminTR is 851 yminRTR 884, ymaxTR: 1110, ymaxRTR: 1125, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 5, Check_Line_distance_with_hla is 94,Check_Line_distance_with_hla_ymin is 103 yminTR is 838 yminRTR 865, ymaxTR: 1099, ymaxRTR: 1106, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 4, Check_Line_distance_with_hla is 90,Check_Line_distance_with_hla_ymin is 97 yminTR is 851 yminRTR 884, ymaxTR: 1110, ymaxRTR: 1125, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 1, Check_Line_distance_with_hla is 97,Check_Line_distance_with_hla_ymin is 99 yminTR is 867 yminRTR 896, ymaxTR: 1128, ymaxRTR: 1145, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 2, Check_Line_distance_with_hla is 96,Check_Line_distance_with_hla_ymin is 99 yminTR is 887 yminRTR 909, ymaxTR: 1150, ymaxRTR: 1157, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 2, Check_Line_distance_with_hla is 96,Check_Line_distance_with_hla_ymin is 99 yminTR is 887 yminRTR 909, ymaxTR: 1150, ymaxRTR: 1157, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21
                    #Right E4,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 1, Check_Line_distance_with_hla is 97,Check_Line_distance_with_hla_ymin is 99 yminTR is 867 yminRTR 896, ymaxTR: 1128, ymaxRTR: 1145, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 21


                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_2) or SizeOf_cy_line > 19 or Check_Line_distance_with_hla > 98 or Check_Line_distance_with_hla_ymin < 80:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (500, 700), (775, 1040), color, 12)
                    logger.debug("right E4 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"E4 Exception in Right Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E4 Exception in Right Side check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)  

        #======================================= E3
        if Position == "6":  # E3 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}
            hla_label_dict_left_line = {}
            hla_label_dict_right_line = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 500: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_left_line['2'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        if smalllabellist[1] < 950:
                            hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_right_line['2'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]
                    

                    hla_top_L_line = hla_label_dict_left_line.get('2')
                    if hla_top_L_line is None:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    xminTL_line, yminTL_line, xmaxTL_line, ymaxTL_line,result, cx_line, cy_line = hla_top_L_line[1:8]

                    thresholdPixel_LeftSide_1 = 21
                    thresholdPixel_LeftSide_2 = 80#30
                    SizeOf_cy_line = cy_line -cy  # ok 8
                    Check_Line_distance_with_hla = ymaxTL - ymaxTL_line 
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL 

                    logger.debug(f"Left E3,LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is {SizeOf_cy_line} Check_Line_distance_with_hla is {Check_Line_distance_with_hla} , Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    
                    #Left E3,LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 22 Check_Line_distance_with_hla is 73,Check_Line_distance_with_hla_ymin is 116 Check_Line_distance_with_hla is 73, Check_Line_distance_with_hla_ymin is 116 ymaxTL: 508,yminTL 253 yminRTL 221 ymaxRTL: 464, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 80
                    #Left E3,LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 22 ,Check_Line_distance_with_hla_ymin is 116 Check_Line_distance_with_hla is 73, ymaxTL: 508,yminTL 253 yminRTL 221 ymaxRTL: 464, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 80

              
                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or SizeOf_cy_line > 23 or Check_Line_distance_with_hla > 93 or Check_Line_distance_with_hla_ymin < 80:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (485, 170), (770, 490), color, 12)
                    logger.debug("E3 HLA_TOP or HLA_ROD_TOP_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E3 Exception in Left_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E3 Exception in Left_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)

            # Right Side check
            try:  
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    hla_top_R_line = hla_label_dict_right_line.get('2')
                    if hla_top_R_line is None:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTR_line, yminTR_line, xmaxTR_line, ymaxTR_line,result,cx_line, cy_line = hla_top_R_line[1:8]

                    thresholdPixel_RightSide_1 = 40#30
                    thresholdPixel_RightSide_2 = 20
                    SizeOf_cy_line = cy_line -cy   # ok 8
                    Check_Line_distance_with_hla = ymaxTR - ymaxTR_line 
                    Check_Line_distance_with_hla_ymin = yminTR_line - yminTR 

                    logger.debug(f"Right E3, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is {SizeOf_cy_line} Check_Line_distance_with_hla is {Check_Line_distance_with_hla},Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR},"
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right E3, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 2 Check_Line_distance_with_hla is 85,Check_Line_distance_with_hla_ymin is 90 yminTR is 825 yminRTR 852, ymaxTR: 1079, ymaxRTR: 1103,thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 20
                    #Right E3, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 12 Check_Line_distance_with_hla is 78,Check_Line_distance_with_hla_ymin is 103 yminTR is 818 yminRTR 850, ymaxTR: 1076, ymaxRTR: 1093,thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 20
                    #Right E3, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 3 Check_Line_distance_with_hla is 95,Check_Line_distance_with_hla_ymin is 100 yminTR is 835 yminRTR 860, ymaxTR: 1096, ymaxRTR: 1109,thresholdPixel_RightSide_1: 40, thresholdPixel_RightSide_2: 20
                    #Right E3, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 0 Check_Line_distance_with_hla is 97,Check_Line_distance_with_hla_ymin is 97 yminTR is 825 yminRTR 845, ymaxTR: 1083, ymaxRTR: 1089,thresholdPixel_RightSide_1: 40, thresholdPixel_RightSide_2: 20


                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_2) or SizeOf_cy_line > 21 or Check_Line_distance_with_hla > 98 or Check_Line_distance_with_hla_ymin < 80: 
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (490, 685), (770, 1010), color, 12)
                    logger.debug("E3 HLA_TOP or HLA_ROD_TOP_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 missing on the right side.")

            except Exception as e:
                logger.debug(f"E3 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E3 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)  

        #======================================= E2
        if Position == "7":  # E2 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}
            hla_label_dict_left_line = {}
            hla_label_dict_right_line = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 430: #500
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_left_line['2'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        if smalllabellist[1] < 950:
                            hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_right_line['2'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    hla_top_L_line = hla_label_dict_left_line.get('2')
                    if hla_top_L_line is None:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTL_line, yminTL_line, xmaxTL_line, ymaxTL_line,result, cx_line, cy_line = hla_top_L_line[1:8]

                    thresholdPixel_LeftSide_1 = 21
                    thresholdPixel_LeftSide_2 = 85#70#60#85
                    SizeOf_cy_line = cy_line -cy  #ok 7
                    Check_Line_distance_with_hla = ymaxTL - ymaxTL_line 
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL 

                    logger.debug(f"Left E2,LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is {SizeOf_cy_line} Check_Line_distance_with_hla is {Check_Line_distance_with_hla}, Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    
                    #Left E2,LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 21 Check_Line_distance_with_hla is 69, Check_Line_distance_with_hla_ymin is 110 ymaxTL: 431,yminTL 178 yminRTL 116 ymaxRTL: 355, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 85
                    #Left E2,LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 6 Check_Line_distance_with_hla is 91, Check_Line_distance_with_hla_ymin is 104 ymaxTL: 518,yminTL 250 yminRTL 213 ymaxRTL: 450, thresholdPixel_LeftSide_1: 21, thresholdPixel_LeftSide_2: 85



                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or SizeOf_cy_line > 25 or Check_Line_distance_with_hla > 93 or Check_Line_distance_with_hla_ymin < 80:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                   
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (480, 200), (75, 510), color, 12)
                    logger.debug("E2 HLA_TOP or HLA_ROD_TOP missing_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 on the left side.")
            
            except Exception as e:
                logger.debug(f"E2 Exception in Left Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E2 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    hla_top_R_line = hla_label_dict_right_line.get('2')
                    if hla_top_R_line is None:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTR_line, yminTR_line, xmaxTR_line, ymaxTR_line,result,cx_line, cy_line = hla_top_R_line[1:8]

                    thresholdPixel_RightSide_1 = 30
                    thresholdPixel_RightSide_2 = 35#30#20
                    SizeOf_cy_line = cy_line -cy  #ok 6
                    Check_Line_distance_with_hla = ymaxTR - ymaxTR_line 
                    Check_Line_distance_with_hla_ymin = yminTR_line - yminTR 
              
                    logger.debug(f"Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is {SizeOf_cy_line} Check_Line_distance_with_hla is {Check_Line_distance_with_hla},Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} is yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")

                    #Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is -1 Check_Line_distance_with_hla is 90,Check_Line_distance_with_hla_ymin is 89 is yminTR is 780 yminRTR 781, ymaxTR: 1036, ymaxRTR: 1027, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 20
                    #Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is 4 Check_Line_distance_with_hla is 88,Check_Line_distance_with_hla_ymin is 96 is yminTR is 803 yminRTR 827, ymaxTR: 1060, ymaxRTR: 1068, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 20
                    #Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is 4 Check_Line_distance_with_hla is 88,Check_Line_distance_with_hla_ymin is 95 is yminTR is 860 yminRTR 891, ymaxTR: 1121, ymaxRTR: 1128, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 30
                    #Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is 0 Check_Line_distance_with_hla is 97,Check_Line_distance_with_hla_ymin is 97 is yminTR is 855 yminRTR 875, ymaxTR: 1115, ymaxRTR: 1112, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 35
                    #Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is -3 Check_Line_distance_with_hla is 100,Check_Line_distance_with_hla_ymin is 94 is yminTR is 789 yminRTR 791, ymaxTR: 1044, ymaxRTR: 1039, 
                    #Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is 1 Check_Line_distance_with_hla is 99,Check_Line_distance_with_hla_ymin is 102 is yminTR is 835 yminRTR 841, ymaxTR: 1095, ymaxRTR: 1094, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 35
                    #Right E2,RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 SizeOf_cy_line is 0 Check_Line_distance_with_hla is 102,Check_Line_distance_with_hla_ymin is 102 is yminTR is 829 yminRTR 847, ymaxTR: 1095, ymaxRTR: 1092, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 35


                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_1) or SizeOf_cy_line > 18 or Check_Line_distance_with_hla > 103 or Check_Line_distance_with_hla_ymin < 80:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (490, 715), (740, 1020), color, 12)
                    logger.debug("E2 HLA_TOP or HLA_ROD_TOP_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 missing on the right side.")

            except Exception as e:
                logger.debug(f"E2 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E2 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)  

        #======================================= E1
        if Position == "8":  # E1 position check
            hla_label_dict_left = {}
            hla_label_dict_right = {}
            hla_label_dict_left_line = {}
            hla_label_dict_right_line = {}


            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 600: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_left_line['2'] = smalllabellist
                else:  # Right-side detection
                    
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        if xmax < 950:
                            hla_label_dict_right['0'] = smalllabellist
                      #  hla_label_dict_right['0'] = smalllabellist
                    #elif smalllabellist[0] == 'HLA_ROD_TOP':
                    elif smalllabellist[0] == 'HLA_ROD_TOP' and ymin < 1050:
                        hla_label_dict_right['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_right_line['2'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL,result, Lcx, Lcy = hla_rod_top_L[1:8]

                    hla_top_L_line = hla_label_dict_left_line.get('2')
                    if hla_top_L_line is None:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTL_line, yminTL_line, xmaxTL_line, ymaxTL_line,result, cx_line, cy_line = hla_top_L_line[1:8]

                    
                    thresholdPixel_LeftSide_1 = 20
                    thresholdPixel_LeftSide_2 = 96 #30
                    SizeOf_cy_line = cy_line -cy #ok-11

                    Check_Line_distance_with_hla = ymaxTL - ymaxTL_line #ok 80
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL  #ok=103
                
                    logger.debug(f"Left E1, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is {SizeOf_cy_line} Check_Line_distance_with_hla is {Check_Line_distance_with_hla}, Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} ymaxTL: {ymaxTL},yminTL {yminTL} yminRTL {yminRTL} ymaxRTL: {ymaxRTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    
                    #Left E1, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 6 Check_Line_distance_with_hla is 78, Check_Line_distance_with_hla_ymin is 91 ymaxTL: 426,yminTL 168 yminRTL 96 ymaxRTL: 345, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 90
                    #Left E1, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 4 Check_Line_distance_with_hla is 92, Check_Line_distance_with_hla_ymin is 101 ymaxTL: 462,yminTL 192 yminRTL 140 ymaxRTL: 393, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 90
                    #Left E1, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 4 Check_Line_distance_with_hla is 92, Check_Line_distance_with_hla_ymin is 101 ymaxTL: 462,yminTL 192 yminRTL 140 ymaxRTL: 393, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 90
                    #Left E1, LeftSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 10 Check_Line_distance_with_hla is 86, Check_Line_distance_with_hla_ymin is 106 ymaxTL: 353,yminTL 87 yminRTL 14 ymaxRTL: 260, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 90
       
                    if yminTL < (yminRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or SizeOf_cy_line > 19 or Check_Line_distance_with_hla > 95 or Check_Line_distance_with_hla_ymin < 85:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (275, 22), (460, 330), color, 12)
                    logger.debug("E1 HLA_TOP or HLA_ROD_TOP_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E1 Exception in Left_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E1 Exception in Left_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)

            # Right Side check
            try:              
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR,result,Rcx, Rcy = hla_rod_top_R[1:8]

                    hla_top_R_line = hla_label_dict_right_line.get('2')
                    if hla_top_R_line is None:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                        
                    xminTR_line, yminTR_line, xmaxTR_line, ymaxTR_line,result,cx_line, cy_line = hla_top_R_line[1:8]

                    thresholdPixel_RightSide_1 = 35
                    thresholdPixel_RightSide_2 = 45
                    SizeOf_cy_line = cy_line -cy  #ok -9
                    Check_Line_distance_with_hla = ymaxTR - ymaxTR_line #78
                    Check_Line_distance_with_hla_ymin = yminTR_line - yminTR  # 104

                    logger.debug(f"Right E1, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is {SizeOf_cy_line} Check_Line_distance_with_hla is {Check_Line_distance_with_hla},Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} yminTR is {yminTR} yminRTR {yminRTR}, ymaxTR: {ymaxTR}, ymaxRTR: {ymaxRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right E1, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is 4 Check_Line_distance_with_hla is 85,Check_Line_distance_with_hla_ymin is 92 yminTR is 723 yminRTR 710, ymaxTR: 980, ymaxRTR: 959, thresholdPixel_RightSide_1: 35, thresholdPixel_RightSide_2: 45
                    #Right E1, RightSide_Cet_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is SizeOf_cy_line is -2 Check_Line_distance_with_hla is 96,Check_Line_distance_with_hla_ymin is 92 yminTR is 826 yminRTR 840, ymaxTR: 1093, ymaxRTR: 1097, thresholdPixel_RightSide_1: 35, thresholdPixel_RightSide_2: 45


                    if yminTR < (yminRTR - thresholdPixel_RightSide_2) or ymaxTR > (ymaxRTR + thresholdPixel_RightSide_1) or SizeOf_cy_line > 21 or Check_Line_distance_with_hla > 97 or Check_Line_distance_with_hla_ymin < 85:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (440, 175), (760, 505), color, 12)
                    logger.debug("E1 HLA_TOP or HLA_ROD_TOP_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 missing on the right side.")

            except Exception as e:
                logger.debug(f"E1 Exception in Right Side check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50: {e}")
                print("E1 Exception in Right_Side_check_HLA_PositionCheck_WITH_ROD_4CYLINDER_C50:", e)  

        #=======================================    status
        if HLA_POSTION_LEFT == "OK" and HLA_POSTION_RIGHT == "OK":  
            HLA_POSTION = "OK"
             
        else:
            HLA_POSTION ="NOT OK"

        
    except Exception as e:
        logger.debug(f"except Exception HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 {e}")
        print("HLA_PositionCheck_WITH_ROD_4CYLINDER_C50 is :",e)
        HLA_POSTION ="NOT OK"
        return HLA_POSTION, original_image

    return HLA_POSTION, original_image



def HLA_PositionCheck_WITH_ROD_3CYLINDER(Position, OBJECT_LIST, original_image):
    HLA_POSTION_RIGHT = "NOT OK"
    HLA_POSTION_LEFT = "NOT OK"
    HLA_POSTION = "NOT OK"
    HLA_POSTION_LEFT = "NOT OK"

    try:
        if Position == "1":  # I1 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 400:
                    if (smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL") and ymax > 90: 
                   # if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL" and  ymin > 50:
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL = hla_rod_top_L[1:5]
                    thresholdPixel_LeftSide_1 = 5#1#10
                    thresholdPixel_LeftSide_2 = 70#60  #CHG
                    HlaSize = ymaxTL - yminTL  #ok 232   #nok -185
                    # if ymaxTL > 545:
                    #     thresholdPixel_LeftSide_2 = 58

                    logger.debug(f"Left I1, HlaSize is {HlaSize} yminTL: {yminTL}  ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, xminTL: {xminTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left I1, HlaSize is 268  ymaxTL: 422, ymaxRTL: 367, xminTL: 1146, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 50
                    #Left I1, HlaSize is 252  ymaxTL: 377, ymaxRTL: 349, xminTL: 1186, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 20
                    #Left I1, HlaSize is 259  ymaxTL: 408, ymaxRTL: 346, xminTL: 1302, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 60
                    #Left I1, HlaSize is 251  ymaxTL: 382, ymaxRTL: 379, xminTL: 1321, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 70
                    #Left I1, HlaSize is 251  ymaxTL: 379, ymaxRTL: 376, xminTL: 1316, thresholdPixel_LeftSide_1: 10, thresholdPixel_LeftSide_2: 70

                    if ymaxTL < (ymaxRTL + thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or HlaSize < 200 : #ok204
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                     
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1240, 200), (1470, 55), color, 12)
                    logger.debug("left i1 HLA_TOP or HLA_ROD_TOP missing on the right side.")
            
            except Exception as e:
                logger.debug(f"I1 Exception in Left Side check: {e}")
                print("I1 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR = hla_rod_top_R[1:5]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 60#71#73#58#48#40#30  #CHG
                    thresholdPixel_RightSide_2 = 10
                    HlaSize = ymaxTR - yminTR   #ok 230   #nok -184

                    logger.debug(f"Right I1, HlaSize is {HlaSize}  yminTR: {yminTR}  yminTR: {yminTR}, yminRTR: {yminRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right I1, HlaSize is 257  yminTR: 758, yminRTR: 790, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 10
                    #Right I1, HlaSize is 256  yminTR: 753, yminRTR: 784, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 10
                    #Right I1, HlaSize is 259  yminTR: 739, yminRTR: 792, thresholdPixel_RightSide_1: 48, thresholdPixel_RightSide_2: 10


                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or yminTR > (yminRTR + thresholdPixel_RightSide_2) or HlaSize < 210 :  # 736
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1240, 780), (1470, 1100), color, 12)
                    logger.debug("right i1 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I1 Exception in Right Side check: {e}")
                print("I1 Exception in Right Side check:", e)



        #==========================I2====================================
        if Position == "2":  # I2 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 450: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check
            try:
             
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL = hla_rod_top_L[1:5]
                    thresholdPixel_LeftSide_1 = 20
                    thresholdPixel_LeftSide_2 = 50#56#60#64#56#50 #20
                    HlaSize = ymaxTL - yminTL   #ok 239   # nok -172
                    logger.debug(f"Left I2, HlaSize is {HlaSize}  ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, xminTL: {xminTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left I2, HlaSize is 249  ymaxTL: 321, ymaxRTL: 269, xminTL: 1307, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 50
                    #Left I2, HlaSize is 254  ymaxTL: 362, ymaxRTL: 304, xminTL: 1313, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 56
                    #Left I2, HlaSize is 237  ymaxTL: 400, ymaxRTL: 344, xminTL: 1340, thresholdPixel_LeftSide_1: 20, thresholdPixel_LeftSide_2: 56

                    if ymaxTL < (ymaxRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or HlaSize < 200 :  #228 #ok 204
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1245, 160), (1460, 460), color, 12)
                    logger.debug("left i2 HLA_TOP or HLA_ROD_TOP missing on the right side.")
            
            except Exception as e:
                logger.debug(f"I2 Exception in Left Side check: {e}")
                print("I2 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR = hla_rod_top_R[1:5]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 70#67#60#52#45#40
                    thresholdPixel_RightSide_2 = 10

                    if yminTR < 718:
                        thresholdPixel_RightSide_1 = 55#64

                    # if yminRTR > 750 and yminRTR < 790:
                    HlaSize = ymaxTR - yminTR     #ok 234  #139

                    logger.debug(f"Right I2,HlaSize_3cy is {HlaSize}  yminTR: {yminTR}, yminRTR: {yminRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right I2,HlaSize_3cy is 255  yminTR: 719, yminRTR: 761, thresholdPixel_RightSide_1: 40, thresholdPixel_RightSide_2: 10
                    #Right I2,HlaSize_3cy is 261  yminTR: 772, yminRTR: 821, thresholdPixel_RightSide_1: 45, thresholdPixel_RightSide_2: 10
                    #Right I2,HlaSize_3cy is 258  yminTR: 691, yminRTR: 746, thresholdPixel_RightSide_1: 52, thresholdPixel_RightSide_2: 10
                    #Right I2,HlaSize_3cy is 260  yminTR: 743, yminRTR: 806, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 10
                    #Right I2,HlaSize is 254  yminTR: 760, yminRTR: 823, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 10
                    #Right I2,HlaSize is 254  yminTR: 760, yminRTR: 823, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 10
                    #Right I2,HlaSize is 254  yminTR: 760, yminRTR: 823, thresholdPixel_RightSide_1: 60, thresholdPixel_RightSide_2: 10
                                                           #yminRTR = NOT OK =786
                    #Right I2,HlaSize_3cy is 266  yminTR: 792, yminRTR: 871, thresholdPixel_RightSide_1: 67, thresholdPixel_RightSide_2: 10
                    #Right I2,HlaSize_3cy is 263  yminTR: 832, yminRTR: 901, thresholdPixel_RightSide_1: 67, thresholdPixel_RightSide_2: 10


                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or yminTR > (yminRTR + thresholdPixel_RightSide_2) or HlaSize < 210 :   #737 #ok 198
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1240, 740), (1460, 1040), color, 12)
                    logger.debug("i2 right HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"I2 Exception in Right Side check: {e}")
                print("I2 Exception in Right Side check:", e)

        #==========================I3====================================
        if Position == "3":  # I3 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}
            HLA_POSTION_RIGHT = "NOT OK"
            HLA_POSTION_LEFT = "NOT OK"

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 450: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist

            # Left Side check   NOT OK
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL = hla_rod_top_L[1:5]
                    thresholdPixel_LeftSide_1 = 55#80 #100 #CHG
                    thresholdPixel_LeftSide_2 = 53#60#40#30
                    HlaSize = ymaxTL - yminTL     #ok 229  # nok -139
                    logger.debug(f"Left I3, HlaSize is {HlaSize}  ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, xminTL: {xminTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left I3, HlaSize is 228  ymaxTL: 327, ymaxRTL: 293, xminTL: 1076, thresholdPixel_LeftSide_1: 100, thresholdPixel_LeftSide_2: 30
                    #Left I3, HlaSize is 240  ymaxTL: 345, ymaxRTL: 302, xminTL: 1169, thresholdPixel_LeftSide_1: 100, thresholdPixel_LeftSide_2: 40
                    #Left I3, HlaSize is 269  ymaxTL: 385, ymaxRTL: 334, xminTL: 1206, thresholdPixel_LeftSide_1: 100, thresholdPixel_LeftSide_2: 50
                    #Left I3, HlaSize is 62  ymaxTL: 369, ymaxRTL: 537, xminTL: 1122, thresholdPixel_LeftSide_1: 100, thresholdPixel_LeftSide_2: 60


                    if ymaxTL < (ymaxRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or HlaSize < 195 :#265 #ok 200
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                     
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"I3 HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1110, 180), (1260, 460), color, 12)
                    logger.debug("i3 left HLA_TOP or HLA_ROD_TOP missing on the right side.")

            
            except Exception as e:
                logger.debug(f"I3 Exception in Left Side check: {e}")
                print("I3 Exception in Left Side check:", e)

            # Right Side check
            try:              
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR = hla_rod_top_R[1:5]

                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 50#70
                    thresholdPixel_RightSide_2 = 20
                    HlaSize = ymaxTR - yminTR    # #ok 219
                    logger.debug(f"Right I3,HlaSize is {HlaSize} yminTR: {yminTR}, yminRTR: {yminRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")

                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or yminTR > (yminRTR + thresholdPixel_RightSide_2) or HlaSize < 190: #191
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"I3 HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (1085, 690), (1290, 980), color, 12)
                    logger.debug("i3 right HLA_TOP or HLA_ROD_TOP missing on the right side.")


            except Exception as e:
                logger.debug(f"I3 Exception in Right Side check: {e}")
                print("I3 Exception in Right Side check:", e)  
 
        #======================================= E3
        if Position == "4":  # E3 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_left_line = {}
            hla_label_dict_right = {}
            hla_label_dict_right_line = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 450: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_left_line['2'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_right_line['2'] = smalllabellist

            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL = hla_rod_top_L[1:5]

                    hla_top_L_line = hla_label_dict_left_line.get('2')
                    if hla_top_L_line is None:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTL_line, yminTL_line, xmaxTL_line, ymaxTL_line,result, cx_line, cy_line = hla_top_L_line[1:8]
                    thresholdPixel_LeftSide_1 = 80#70
                    thresholdPixel_LeftSide_2 = 70#55#40#55
                    SizeOf_cy_line = cy_line - cy # Ensures the result is always positive ok-11
                    Check_Line_distance_with_hla = ymaxTL - ymaxTL_line  # ok 73
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL #ok = 101 # 84notok  #OK-87

                    logger.debug(f"Left E3,Check_Line_distance_with_hla is {Check_Line_distance_with_hla} Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} SizeOf_cy_line is {SizeOf_cy_line} ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, xminTL: {xminTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left E3,cy is 341 SizeOf_cy_line is 4 ymaxTL: 469, ymaxRTL: 442, xminTL: 491, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 55
                    #Left E3,cy is 393 SizeOf_cy_line is -3 ymaxTL: 521, ymaxRTL: 512, xminTL: 535, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 55
                    #Left E3,cy is 385 SizeOf_cy_line is 13 ymaxTL: 511, ymaxRTL: 497, xminTL: 469, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 55
                    #Left E3,cy is 297 SizeOf_cy_line is 18 ymaxTL: 429, ymaxRTL: 396, xminTL: 521, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 55
                    #Left E3,Check_Line_distance_with_hla is 73 SizeOf_cy_line is 9 ymaxTL: 353, ymaxRTL: 311, xminTL: 592, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 40
                    #Left E3,Check_Line_distance_with_hla is 78 SizeOf_cy_line is 8 ymaxTL: 411, ymaxRTL: 364, xminTL: 576, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 45
                    #Left E3,Check_Line_distance_with_hla is 71 SizeOf_cy_line is 11 ymaxTL: 306, ymaxRTL: 248, xminTL: 606, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 55
                    #Left E3,Check_Line_distance_with_hla is 68 Check_Line_distance_with_hla_ymin is 106 SizeOf_cy_line is 19 ymaxTL: 368, ymaxRTL: 341, xminTL: 580, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 70
                    #Left E3,Check_Line_distance_with_hla is 68 Check_Line_distance_with_hla_ymin is 106 SizeOf_cy_line is 19 ymaxTL: 368, ymaxRTL: 341, xminTL: 580, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 70
                    #Left E3,Check_Line_distance_with_hla is 78 Check_Line_distance_with_hla_ymin is 87 SizeOf_cy_line is 5 ymaxTL: 490, ymaxRTL: 468, xminTL: 622, thresholdPixel_LeftSide_1: 80, thresholdPixel_LeftSide_2: 70


                    if ymaxTL < (ymaxRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or Check_Line_distance_with_hla > 109 or Check_Line_distance_with_hla_ymin < 82 or Check_Line_distance_with_hla_ymin > 144 or SizeOf_cy_line > 18:  #ok-13
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:      
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (490, 190), (745, 530), color, 12)
                    logger.debug("E3 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E3 Exception in Left Side check: {e}")
                print("E3 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR = hla_rod_top_R[1:5]

                    hla_top_R_line = hla_label_dict_right_line.get('2')

                    if hla_top_R_line is None:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    
                    xminTR_line, yminTR_line, xmaxTR_line, ymaxTR_line,result,cx_line, cy_line = hla_top_R_line[1:8]

                    thresholdPixel_RightSide_1 = 72#68#61#49#38#33#37#22#95
                    thresholdPixel_RightSide_2 = 40 #22#12#30

                    SizeOf_cy_line = cy_line - cy  # Ensures the result is always positive
                    Check_Line_distance_with_hla = ymaxTR - ymaxTR_line #ok 94 #NOT-107
                  #  Check_Line_distance_with_hla_ymin = yminTL_line - yminTL #ok = 101
                    Check_Line_distance_with_hla_ymin = yminTR_line - yminTR #ok = 101 -89  #ok 75

                    logger.debug(f"Right E3, Check_Line_distance_with_hla is {Check_Line_distance_with_hla} Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} SizeOf_cy_line {SizeOf_cy_line} yminTR: {yminTR}, yminRTR: {yminRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    
                    ##Right E3, Check_Line_distance_with_hla is 80 Check_Line_distance_with_hla_ymin is 102 SizeOf_cy_line 1 yminTR: 801, yminRTR: 835, thresholdPixel_RightSide_1: 22, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 84 Check_Line_distance_with_hla_ymin is 96 SizeOf_cy_line -2 yminTR: 833, yminRTR: 865, thresholdPixel_RightSide_1: 22, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 85 Check_Line_distance_with_hla_ymin is 101 SizeOf_cy_line -1 yminTR: 858, yminRTR: 897, thresholdPixel_RightSide_1: 37, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 87 Check_Line_distance_with_hla_ymin is 99 SizeOf_cy_line -6 yminTR: 861, yminRTR: 898, thresholdPixel_RightSide_1: 33, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 84 Check_Line_distance_with_hla_ymin is 94 SizeOf_cy_line -7 yminTR: 877, yminRTR: 904, thresholdPixel_RightSide_1: 22, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 83 Check_Line_distance_with_hla_ymin is 97 SizeOf_cy_line 2 yminTR: 791, yminRTR: 833, thresholdPixel_RightSide_1: 38, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 90 Check_Line_distance_with_hla_ymin is 99 SizeOf_cy_line 0 yminTR: 849, yminRTR: 895, thresholdPixel_RightSide_1: 38, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 86 Check_Line_distance_with_hla_ymin is 97 SizeOf_cy_line -2 yminTR: 853, yminRTR: 894, thresholdPixel_RightSide_1: 38, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 85 Check_Line_distance_with_hla_ymin is 100 SizeOf_cy_line 2 yminTR: 784, yminRTR: 840, thresholdPixel_RightSide_1: 49, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 85 Check_Line_distance_with_hla_ymin is 98 SizeOf_cy_line 2 yminTR: 834, yminRTR: 897, thresholdPixel_RightSide_1: 61, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 91 Check_Line_distance_with_hla_ymin is 98 SizeOf_cy_line -2 yminTR: 854, yminRTR: 926, thresholdPixel_RightSide_1: 61, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 90 Check_Line_distance_with_hla_ymin is 89 SizeOf_cy_line -1 yminTR: 925, yminRTR: 1006, thresholdPixel_RightSide_1: 72, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 89 Check_Line_distance_with_hla_ymin is 65 SizeOf_cy_line -12 yminTR: 901, yminRTR: 928, thresholdPixel_RightSide_1: 72, thresholdPixel_RightSide_2: 40
                    #Right E3, Check_Line_distance_with_hla is 100 Check_Line_distance_with_hla_ymin is 55 SizeOf_cy_line -22 yminTR: 942, yminRTR: 996, thresholdPixel_RightSide_1: 72, thresholdPixel_RightSide_2: 40



                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or yminTR > (yminRTR + thresholdPixel_RightSide_2) or Check_Line_distance_with_hla < 80 or Check_Line_distance_with_hla > 103 or Check_Line_distance_with_hla_ymin < 52 or Check_Line_distance_with_hla_ymin > 100 or SizeOf_cy_line > 15:   #or SizeOf_cy_line < 7 or SizeOf_cy_line > 18:   #ok-12
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                  
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"E3 HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (520, 830), (750, 1122), color, 12)
                    logger.debug("E3 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"E4 Exception in Right Side check: {e}")
                print("E4 Exception in Right Side check:", e)  

        #======================================= E2
        if Position == "5":  # E2 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}
            hla_label_dict_left_line = {}
            hla_label_dict_right_line = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]

                if smalllabellist[2] < 450: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        # if smalllabellist[2] < 60:
                        #     logger.debug(f"HLA_ROD_TOP class not valid smalllabellist[2] ",smalllabellist[2])
                        #     continue
                        hla_label_dict_left['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_left_line['2'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                
                        hla_label_dict_right['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_right_line['2'] = smalllabellist

            # Left Side check
            try:
              
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL = hla_rod_top_L[1:5]

                    hla_top_L_line = hla_label_dict_left_line.get('2')
                    if hla_top_L_line is None:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTL_line, yminTL_line, xmaxTL_line, ymaxTL_line,result, cx_line, cy_line = hla_top_L_line[1:8]

                    thresholdPixel_LeftSide_1 = 70
                    thresholdPixel_LeftSide_2 = 69#65#55#45#35
                    SizeOf_cy_line = cy_line - cy # Ensures the result is always positive# ok -27
                    Check_Line_distance_with_hla = ymaxTL - ymaxTL_line  #ok 90,
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL #ok = 101

                    logger.debug(f"Left E2, Check_Line_distance_with_hla is {Check_Line_distance_with_hla} Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} SizeOf_cy_line is {SizeOf_cy_line} ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, xminTL: {xminTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left E2,SizeOf_cy_line is 20 ymaxTL: 340, ymaxRTL: 279, xminTL: 452, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 55
                    #Left E2,SizeOf_cy_line is 8 ymaxTL: 368, ymaxRTL: 321, xminTL: 425, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    #Left E2,SizeOf_cy_line is 24 ymaxTL: 371, ymaxRTL: 336, xminTL: 457, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    #Left E2,SizeOf_cy_line is 27 ymaxTL: 345, ymaxRTL: 305, xminTL: 499, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    #Left E2,SizeOf_cy_line is 225 ymaxTL: 351, ymaxRTL: 309, xminTL: 526, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    #Left E2,SizeOf_cy_line is 225 ymaxTL: 351, ymaxRTL: 309, xminTL: 526, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    #Left E2,SizeOf_cy_line is 3 ymaxTL: 456, ymaxRTL: 416, xminTL: 588, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    # Left E2, Check_Line_distance_with_hla is 88 Check_Line_distance_with_hla_ymin is 100 SizeOf_cy_line is 6 ymaxTL: 340, ymaxRTL: 285, xminTL: 560, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    #Left E2, Check_Line_distance_with_hla is 76 Check_Line_distance_with_hla_ymin is 110 SizeOf_cy_line is 17 ymaxTL: 358, ymaxRTL: 292, xminTL: 625, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 65
                    #Left E2, Check_Line_distance_with_hla is 95 Check_Line_distance_with_hla_ymin is 96 SizeOf_cy_line is 0 ymaxTL: 456, ymaxRTL: 404, xminTL: 663, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 69
                    #Left E2, Check_Line_distance_with_hla is 94 Check_Line_distance_with_hla_ymin is 105 SizeOf_cy_line is 6 ymaxTL: 487, ymaxRTL: 438, xminTL: 640, thresholdPixel_LeftSide_1: 70, thresholdPixel_LeftSide_2: 69


                    if ymaxTL < (ymaxRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or Check_Line_distance_with_hla > 97 or Check_Line_distance_with_hla_ymin < 90 or SizeOf_cy_line > 30:   #ok-16
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:          
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (490, 130), (850, 450), color, 12)
                    logger.debug("E2 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E2 Exception in Left Side check: {e}")
                print("E2 Exception in Left Side check:", e)

            # Right Side check
            try:
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR = hla_rod_top_R[1:5]

                    hla_top_R_line = hla_label_dict_right_line.get('2')
                    if hla_top_R_line is None:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTR_line, yminTR_line, xmaxTR_line, ymaxTR_line,result,cx_line, cy_line = hla_top_R_line[1:8]
                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 65#55#40#28#70
                    thresholdPixel_RightSide_2 = 37#30#24#20

                    #Right E2, Check_Line_distance_with_hla is 88 Check_Line_distance_with_hla_ymin is 100 SizeOf_cy_line is -2 yminTR: 676, yminRTR: 695, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 92 Check_Line_distance_with_hla_ymin is 79 SizeOf_cy_line is -6 yminTR: 800, yminRTR: 835, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 87 Check_Line_distance_with_hla_ymin is 81 SizeOf_cy_line is -3 yminTR: 778, yminRTR: 791, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 96 Check_Line_distance_with_hla_ymin is 82 SizeOf_cy_line is -7 yminTR: 800, yminRTR: 831, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 88 Check_Line_distance_with_hla_ymin is 78 SizeOf_cy_line is -5 yminTR: 777, yminRTR: 795, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 90 Check_Line_distance_with_hla_ymin is 73 SizeOf_cy_line is -9 yminTR: 823, yminRTR: 859, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 91 Check_Line_distance_with_hla_ymin is 74 SizeOf_cy_line is -9 yminTR: 777, yminRTR: 794, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37

                    #Right E2, Check_Line_distance_with_hla is 94 Check_Line_distance_with_hla_ymin is 71 SizeOf_cy_line is -12 yminTR: 797, yminRTR: 810, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 99 Check_Line_distance_with_hla_ymin is 75 SizeOf_cy_line is -12 yminTR: 780, yminRTR: 802, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 90 Check_Line_distance_with_hla_ymin is 91 SizeOf_cy_line is 0 yminTR: 844, yminRTR: 878, thresholdPixel_RightSide_1: 28, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 92 Check_Line_distance_with_hla_ymin is 77 SizeOf_cy_line is -7 yminTR: 840, yminRTR: 875, thresholdPixel_RightSide_1: 28, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 94 Check_Line_distance_with_hla_ymin is 86 SizeOf_cy_line is -4 yminTR: 807, yminRTR: 850, thresholdPixel_RightSide_1: 40, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 85 Check_Line_distance_with_hla_ymin is 89 SizeOf_cy_line is 2 yminTR: 806, yminRTR: 841, thresholdPixel_RightSide_1: 28, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 98 Check_Line_distance_with_hla_ymin is 82 SizeOf_cy_line is -8 yminTR: 868, yminRTR: 924, thresholdPixel_RightSide_1: 55, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 92 Check_Line_distance_with_hla_ymin is 78 SizeOf_cy_line is -7 yminTR: 884, yminRTR: 919, thresholdPixel_RightSide_1: 65, thresholdPixel_RightSide_2: 37
                    #Right E2, Check_Line_distance_with_hla is 103 Check_Line_distance_with_hla_ymin is 79 SizeOf_cy_line is -12 yminTR: 767, yminRTR: 794, thresholdPixel_RightSide_1: 65, thresholdPixel_RightSide_2: 37


                    SizeOf_cy_line = cy_line - cy# Ensures the result is always positive        
                    Check_Line_distance_with_hla = ymaxTR - ymaxTR_line # ok 88
                    Check_Line_distance_with_hla_ymin = yminTR_line - yminTR #ok = 101 -89  #ok 75
                    logger.debug(f"Right E2, Check_Line_distance_with_hla is {Check_Line_distance_with_hla} Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} SizeOf_cy_line is {SizeOf_cy_line} yminTR: {yminTR}, yminRTR: {yminRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")

                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or yminTR > (yminRTR + thresholdPixel_RightSide_2) or Check_Line_distance_with_hla < 82 or Check_Line_distance_with_hla > 106 or Check_Line_distance_with_hla_ymin < 70 or Check_Line_distance_with_hla_ymin > 97 or SizeOf_cy_line > 15:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:        
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (500, 740), (820, 1060), color, 12)
                    logger.debug("E2 HLA_TOP or HLA_ROD_TOP missing on the945 right side.")

            except Exception as e:
                logger.debug(f"E2 Exception in Right Side check: {e}")
                print("E2 Exception in Right Side check:", e)  

        #======================================= E1
        if Position == "6":  # E1 position check
            leftSide_Cord = 0
            RightSide_Cord = 0
            hla_label_dict_left = {}
            hla_label_dict_right = {}
            hla_label_dict_left_line = {}
            hla_label_dict_right_line = {}

            # Iterate over object list to classify based on position
            for smalllabellist in OBJECT_LIST:
                xmin, ymin, xmax, ymax = smalllabellist[1:5]
                #print(smalllabellist[0])

                if smalllabellist[2] < 420:#500: 
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_left['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_left['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_left_line['2'] = smalllabellist
                else:  # Right-side detection
                    if smalllabellist[0] == 'HLA_TOP' or smalllabellist[0] == "HLA_3CYL":
                        hla_label_dict_right['0'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_ROD_TOP':
                        hla_label_dict_right['1'] = smalllabellist
                    elif smalllabellist[0] == 'HLA_3CYL_LINE':
                        hla_label_dict_right_line['2'] = smalllabellist

            # Left Side check
            try:
                if '0' in hla_label_dict_left and '1' in hla_label_dict_left:
                    hla_top_L = hla_label_dict_left.get('0')
                    xminTL, yminTL, xmaxTL, ymaxTL,result, cx, cy = hla_top_L[1:8]

                    hla_rod_top_L = hla_label_dict_left.get('1')
                    xminRTL, yminRTL, xmaxRTL, ymaxRTL = hla_rod_top_L[1:5]

                    hla_top_L_line = hla_label_dict_left_line.get('2')
                    if hla_top_L_line is None:
                        logger.debug(f"HLA_POSTION_LEFT = NOT OK")
                        HLA_POSTION_LEFT = "NOT OK"  # HAL HLA
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTL_line, yminTL_line, xmaxTL_line, ymaxTL_line,result, cx_line, cy_line = hla_top_L_line[1:8]

                    thresholdPixel_LeftSide_1 = 45#70#90
                    thresholdPixel_LeftSide_2 = 81#76#70

                    SizeOf_cy_line = cy_line - cy # Ensures the result is always positive
                    Check_Line_distance_with_hla = ymaxTL - ymaxTL_line #73
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL #ok = 101
                    logger.debug(f"Left E1,Check_Line_distance_with_hla is {Check_Line_distance_with_hla} Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin}  SizeOf_cy_line is {SizeOf_cy_line} ymaxTL: {ymaxTL}, ymaxRTL: {ymaxRTL}, xminTL: {xminTL}, "
                                 f"thresholdPixel_LeftSide_1: {thresholdPixel_LeftSide_1}, "
                                 f"thresholdPixel_LeftSide_2: {thresholdPixel_LeftSide_2}")
                    #Left E1,cy is 319 SizeOf_cy_line is 5 ymaxTL: 451, ymaxRTL: 422, xminTL: 561, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 70
                    # # Left E1,Check_Line_distance_with_hla is 83  SizeOf_cy_line is 9 ymaxTL: 371, ymaxRTL: 300, xminTL: 601, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 70
                    #Left E1,Check_Line_distance_with_hla is 77 Check_Line_distance_with_hla_ymin is 123  SizeOf_cy_line is 23 ymaxTL: 469, ymaxRTL: 418, xminTL: 645, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 76
                    #Left E1,Check_Line_distance_with_hla is 73 Check_Line_distance_with_hla_ymin is 129  SizeOf_cy_line is 28 ymaxTL: 479, ymaxRTL: 432, xminTL: 591, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 76
                    #Left E1,Check_Line_distance_with_hla is 94 Check_Line_distance_with_hla_ymin is 97  SizeOf_cy_line is 1 ymaxTL: 523, ymaxRTL: 486, xminTL: 568, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 76
                    #Left E1,Check_Line_distance_with_hla is 81 Check_Line_distance_with_hla_ymin is 109  SizeOf_cy_line is 14 ymaxTL: 421, ymaxRTL: 343, xminTL: 520, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 76
                    #Left E1,Check_Line_distance_with_hla is 97 Check_Line_distance_with_hla_ymin is 102  SizeOf_cy_line is 3 ymaxTL: 482, ymaxRTL: 428, xminTL: 630, thresholdPixel_LeftSide_1: 45, thresholdPixel_LeftSide_2: 81


                    if ymaxTL < (ymaxRTL - thresholdPixel_LeftSide_1) or ymaxTL > (ymaxRTL + thresholdPixel_LeftSide_2) or Check_Line_distance_with_hla > 98 or Check_Line_distance_with_hla_ymin < 90 or Check_Line_distance_with_hla_ymin > 120 or SizeOf_cy_line > 30:#Check_Line_distance_with_hla 97
                        HLA_POSTION_LEFT = "NOT OK"
                        logger.debug(f"E1 HLA_POSTION_LEFT =NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTL, yminTL), (xmaxTL, ymaxTL), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTL, yminTL - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:                  
                        HLA_POSTION_LEFT = "OK"
                        logger.debug(f"E1 HLA_POSTION_LEFT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (490, 170), (805, 490), color, 12)
                    logger.debug("E1 HLA_TOP or HLA_ROD_TOP missing on the left side.")
            
            except Exception as e:
                logger.debug(f"E1 Exception in Left Side check: {e}")
                print("E1 Exception in Left Side check:", e)

            try:      
                if '0' in hla_label_dict_right and '1' in hla_label_dict_right:
                    hla_top_R = hla_label_dict_right.get('0')
                    xminTR, yminTR, xmaxTR, ymaxTR,result,cx, cy = hla_top_R[1:8]

                    hla_rod_top_R = hla_label_dict_right.get('1')
                    xminRTR, yminRTR, xmaxRTR, ymaxRTR = hla_rod_top_R[1:5]

                    hla_top_R_line = hla_label_dict_right_line.get('2')
                    if hla_top_R_line is None:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)

                    xminTR_line, yminTR_line, xmaxTR_line, ymaxTR_line,result,cx_line, cy_line = hla_top_R_line[1:8]
                    RightSide_Cord = xminRTR - xminTR
                    thresholdPixel_RightSide_1 = 58#50#38#36#30#70
                    thresholdPixel_RightSide_2 = 15#1

                    SizeOf_cy_line = cy_line - cy # Ensures the result is always positive#ok -3
                    Check_Line_distance_with_hla = ymaxTR - ymaxTR_line
                    Check_Line_distance_with_hla_ymin = yminTL_line - yminTL #ok = 101
                    logger.debug(f"Right E1,Check_Line_distance_with_hla is {Check_Line_distance_with_hla} Check_Line_distance_with_hla_ymin is {Check_Line_distance_with_hla_ymin} SizeOf_cy_line is {SizeOf_cy_line}, yminTR: {yminTR}, yminRTR: {yminRTR}, "
                                 f"thresholdPixel_RightSide_1: {thresholdPixel_RightSide_1}, "
                                 f"thresholdPixel_RightSide_2: {thresholdPixel_RightSide_2}")
                    #Right E1,Check_Line_distance_with_hla is 76 SizeOf_cy_line is 19, yminTR: 701, yminRTR: 721, thresholdPixel_RightSide_1: 70, thresholdPixel_RightSide_2: 15
                    #Right E1,Check_Line_distance_with_hla is 88 Check_Line_distance_with_hla_ymin is 106 SizeOf_cy_line is -1, yminTR: 801, yminRTR: 834, thresholdPixel_RightSide_1: 30, thresholdPixel_RightSide_2: 15
                    #Right E1,Check_Line_distance_with_hla is 85 Check_Line_distance_with_hla_ymin is 101 SizeOf_cy_line is 7, yminTR: 746, yminRTR: 793, thresholdPixel_RightSide_1: 36, thresholdPixel_RightSide_2: 15
                    #Right E1,Check_Line_distance_with_hla is 87 Check_Line_distance_with_hla_ymin is 93 SizeOf_cy_line is 1, yminTR: 864, yminRTR: 917, thresholdPixel_RightSide_1: 50, thresholdPixel_RightSide_2: 15


                    if yminTR < (yminRTR - thresholdPixel_RightSide_1) or yminTR > (yminRTR + thresholdPixel_RightSide_2) or Check_Line_distance_with_hla > 115 or Check_Line_distance_with_hla_ymin < 78 or Check_Line_distance_with_hla_ymin > 124 or SizeOf_cy_line > 22:
                        HLA_POSTION_RIGHT = "NOT OK"
                        logger.debug(f"HLA_POSTION_RIGHT = NOT OK")
                        color = (0, 0, 255)
                        cv2.rectangle(original_image, (xminTR, yminTR), (xmaxTR, ymaxTR), color, 12)
                        label_text = ""
                        cv2.putText(original_image, label_text, (xminTR, yminTR - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 7)
                    else:      
                        HLA_POSTION_RIGHT = "OK"
                        logger.debug(f"HLA_POSTION_RIGHT = OK")
                else:
                    color = (0, 0, 255)
                    cv2.rectangle(original_image, (480, 770), (805, 1090), color, 12)
                    logger.debug("E1 HLA_TOP or HLA_ROD_TOP missing on the right side.")

            except Exception as e:
                logger.debug(f"E1 Exception in Right Side check: {e}")
                print("E1 Exception in Right Side check:", e)     

        #=======================================    status
        if HLA_POSTION_LEFT == "OK" and HLA_POSTION_RIGHT == "OK":
            HLA_POSTION = "OK"
             
        else:
            HLA_POSTION ="NOT OK"

        
    except Exception as e:
        logger.debug(f"except Exception HLA_PositionCheck_WITH_ROD_3CYLINDER {e}")
        print("HLA_PositionCheck_WITH_ROD_3CYLINDER is :",e)
        HLA_POSTION ="NOT OK"
        return HLA_POSTION, original_image

    return HLA_POSTION, original_image





def drawPolygonPoints(image,data_list):
    # Set the polygon points
    points = np.array(data_list, np.int32)
    # Reshape the points array to match the required format for cv2.polylines()
    points = points.reshape((-1, 1, 2))
    # Set the color and thickness of the polygon outline
    color = (0, 255, 0)  # BGR color format (red)
    thickness = 5
    # Draw the polygon on the image
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

def getMinMaxValues(values):
    minX = float('inf')
    minY = float('inf')
    maxX = float('-inf')
    maxY = float('-inf')
    
    for point in values:
        x, y = point
        if x < minX:
            minX = x
        if y < minY:
            minY = y
        if x > maxX:
            maxX = x
        if y > maxY:
            maxY = y
    return minX,minY,maxX,maxY


#============================== pdf =========================#
def start_pdf(final_status,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT, engine_number, dirname):
    pdfFile = []
    try:
        #engine_number = "VK1234567890"#Part_number#getEngineNumberDetails()
        print(engine_number)
        TodaysDate = datetime.datetime.now().strftime('%Y_%m_%d')
        #dirName = "/home/viks/VIKS/CODE/PROJECT_ALGORITHAM/NEW_EGAL_PROJECT/pdfgenerate/DEFECT_IMAGES/2023_08_25/"
        dirName =dirname+"/"#f"/home/deepak/Desktop/TOX/OP_40/INF_IMAGES"

        #image_folder_link1 = os.path.join(dirName, TodaysDate)
        #image_folder_link=(dirName+"/"+engine_number)+"/"
        #print(image_folder_link)
        #Status = engine_number
        INF_IMAGE_PATH_LIST=os.listdir(dirName)
        INF_IMAGE_PATH_LIST.sort()
        image_file_list = []
        for image_file in INF_IMAGE_PATH_LIST:
            if image_file.lower().endswith(('.jpg','.jpeg')):
                INF_FOLDER_PATH=dirName+image_file
                image_file_list.append(INF_FOLDER_PATH)
                # Replace double slashes with a single slash in each element of the list
                image_file_list = [path.replace("//", "/") for path in image_file_list]
        
        DateTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        SavedirName = dirName
        pdfFile =createPDF(image_file_list,final_status, OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT,engine_number, DateTime, SavedirName)
        print("createPDF Completed++++++++++++++++++++++")   
        return pdfFile
    except Exception as e:
        print("except Exception in start_pdf",e)
        return pdfFile

def insert_process_data(enginenumber,pdf_file,final_status):
    try:
        dbConnection = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = dbConnection.cursor()
        starttime=datetime.datetime.now().strftime("%Y-%m-%d %H:%S:%M")
        query=f"INSERT INTO `hla_cam_cap_db`.`cam_cap_temp_pdf_path_table`(`cam_cap_temp_pdf_path_table`.`PDF_FILE_PATH`,`cam_cap_temp_pdf_path_table`.`STATUS`,`cam_cap_temp_pdf_path_table`.`ENGINE_NUMBER`) VALUES('{pdf_file}','{final_status}','{enginenumber}');"
        cur.execute(query)
        dbConnection.commit()
        cur.close()
        print("insert the data")
    except Exception as e:
        print(e)
    finally:
        dbConnection.close()

def reduceImageQuality(image_file_list):
    for image in image_file_list:
        image_file = Image.open(image)
        image_file.save(image, quality=25)

def createPDF(image_file_list,final_status, OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT, engine_number,DateTime,SavedirName):
    try:
        global gMailObj
        pdffilename = []
        pdf = FPDF()
        reduceImageQuality(image_file_list)
        pdf.add_page("L")
        pdf.rect(5.0,4.0,285.0,200.0)
        pdf.set_font('Arial', 'B', 30)
        pdf.cell(270,10,"DRISHTI PORTAL",0,1,'C')
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(270,15,"HLA CAM CAP REPORT",0,1,'C')

        pdf.image("D:/INSIGHTZZ/PRODUCATION_CODE/HLA_UI/LOGO/Mahindra-Mahindra-New-Logo.png", x=10, y=5, w=50, h=30, type='png')
        #pdf.image("c:/INSIGHTZZ/PISTON_DROPING_UI_MAIN/LOGO/download.jpg", x=220, y=5, w=60, h=30, type='jpg')

        pdf.rect(5.0,38.0,285.0,166.0)
        pdf.cell(270,30,"Engine Number : "+engine_number,0,1)
        pdf.set_font('Arial', 'B', 15)
        pdf.cell(270,10,"Date & Time of Inspection : "+DateTime,0,1)
        pdf.cell(270,10,"Engine Status : "+final_status,0,1)
        pdf.cell(270,10,"Engine overall cam cap status: "+OVERALL_CAM_CAP_RESULT,0,1)
        pdf.cell(270,10,"Engine hla status : "+OVERALL_HLA_RESULT,0,1)
        pdf.set_font("Times", size=13)   
        image_file_list1=[]
        image__file_list2=[]
        counter=0
        counter2=0
        for image in image_file_list:
            image_file_list1.append(image)
        
        if len(image_file_list1) >0:
            for image in image_file_list1:
                pdf.add_page("L")
                pdf.rect(5.0,4.0,285.0,200.0)
                fixed_height = 170
                pdf.set_font('Arial', 'B', 20)
                pdf.cell(250,10,"IMAGE NAME : "+os.path.basename(image),0,1,'C')
                img = Image.open(image)
                height_percent = (fixed_height / float(img.size[1]))
                width_size = int((float(img.size[0]) * float(height_percent)))
                pdf.image(image,20,20,width_size,fixed_height)
                counter=counter+1
                print(counter)
                if counter==25:
                    break
        
        pdf.output(SavedirName+"/"+engine_number+".pdf", "F")   
        pdffilename=SavedirName+"/"+engine_number+".pdf"
        #insert_process_data(engine_number,pdffilename,final_status)
        return pdffilename
    except Exception as e:
        print("except Exception in createPDF",e)
        return pdffilename


def Update_ClearImageResult(IS_PROCESS, OverallOk, CamCapstatus, HLA_status):
    db_update = None
    cur = None
    try:
        db_update = pymysql.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
        cur = db_update.cursor()
        query = """
            UPDATE engine_status_config 
            SET IS_PROCESS_INF = %s, 
                OVREALL_STATUS = %s, 
                OVERALL_CAM_CAP_RESULT = %s, 
                OVERALL_HLA_RESULT = %s 
            WHERE ID = 1
        """
        cur.execute(query, (IS_PROCESS, OverallOk, CamCapstatus, HLA_status))
        db_update.commit()
        print("================================== Update_ClearImageResult executed successfully.=================")
        logger.debug("Update_ClearImageResult executed successfully.")
    except Exception as e:
        print(f"Update_ClearImageResult is {e}")
        logger.error(f"Exception in Update_ClearImageResult: {str(e)}")
    finally:
        if cur is not None:
            cur.close()
        if db_update is not None:
            db_update.close()


def run(maskRCNNObj):
    #======================================== 4 CYLINDER =======================================#
    CLASS_NAME_OK_LIST_I2_4C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_I3_4C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_I4_4C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_I5_4C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E4_4C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E3_4C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E2_4C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E1_4C = ["HALF_HLA_BASE","HLA_BASE"]

    DEFECT_LIST_I2_4C = ["cam_cap_I","cam_cap_1","cam_cap_3","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_I3_4C = ["cam_cap_I","cam_cap_1","cam_cap_2","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_I4_4C = ["cam_cap_I","cam_cap_1","cam_cap_2","cam_cap_3", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_I5_4C = ["cam_cap_E","cam_cap_1","cam_cap_2","cam_cap_3", "cam_cap_4","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    
    DEFECT_LIST_E4_4C = ["cam_cap_E","cam_cap_1","cam_cap_2","cam_cap_3", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_E3_4C = ["cam_cap_E","cam_cap_1","cam_cap_2","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_E2_4C = ["cam_cap_E","cam_cap_1","cam_cap_3","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_E1_4C = ["cam_cap_E","cam_cap_2","cam_cap_3","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]

    #======================================== 3 CYLINDER =======================================#
    CLASS_NAME_OK_LIST_I1_3C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_I2_3C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_I3_3C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_I4_3C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E4_3C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E3_3C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E2_3C = ["HALF_HLA_BASE","HLA_BASE"]

    CLASS_NAME_OK_LIST_E1_3C = ["HALF_HLA_BASE","HLA_BASE"]

    DEFECT_LIST_I1_3C = ["cam_cap_I","cam_cap_1","cam_cap_3","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_I2_3C = ["cam_cap_I","cam_cap_1","cam_cap_2","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_I3_3C = ["cam_cap_I","cam_cap_1","cam_cap_2","cam_cap_3", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    
    DEFECT_LIST_E3_4C = ["cam_cap_E","cam_cap_1","cam_cap_2","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_E2_4C = ["cam_cap_E","cam_cap_1","cam_cap_3","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]
    DEFECT_LIST_E1_4C = ["cam_cap_E","cam_cap_2","cam_cap_3","cam_cap_4", "cam_cap_5","cam_cap_missing", "hla_base_half", "hla_base", "nut_missing", "hla_base" ,"NOT_OK_E","NOT_OK_arrow","NOT_OK_2","NOT_OK_I","NOT_OK_3"]

    CAM_IMAGE = "D:/INSIGHTZZ/PRODUCATION_CODE/ALGORITHAM/IMG/"
    if DEMO_RUN_FLAG is True:
        CAM_IMAGE = "/home/viks/VIKS/PROJECT_ALGORITHAM/MAHINDRA_CHKHAN_GATE-12/HLA_CAM_CAP/SEP_26/ALGORITHAM/IMG/"
    img_list = sorted([img for img in os.listdir(CAM_IMAGE) if img.endswith(".jpg")])
    if DEMO_RUN_FLAG is False:
        if len(img_list) != 0:
            for i in img_list :
                if "POS_1.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))
                elif "POS_2.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))
                elif "POS_3.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))
                elif "POS_4.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))
                                            
                elif "POS_5.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))
                elif "POS_6.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))
                elif "POS_7.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))
                elif "POS_8.jpg" in i:
                    os.remove(os.path.join(CAM_IMAGE,i))

                print("========Remove IMG, PROCESS DONE==========") 


    try:
        while True:
            imageList = []
            
            img_list = sorted([img for img in os.listdir(CAM_IMAGE) if img.endswith(".jpg")])
            if len(img_list) == 0:
                print("===================== algo image not found=====================")
                time.sleep(0.1)
                continue

            
            class_check_cam1 = False

            defect_list = []
            ok_list = []

            STATUS = "NOT OK"

            HLA_PostionCheck =  "NOT OK"
            Arrow_PostionCheck = False

            CAM_I1_IMAGE_LINK = ""
            CAM_I2_IMAGE_LINK = ""
            CAM_I3_IMAGE_LINK = ""
            CAM_I4_IMAGE_LINK = ""
            CAM_I5_IMAGE_LINK = ""
            CAM_E4_IMAGE_LINK = ""
            CAM_E3_IMAGE_LINK = ""
            CAM_E2_IMAGE_LINK = ""
            CAM_E1_IMAGE_LINK = ""

            POS_3_I3_STATUS = "NOT OK"
            POS_1_I1_STATUS = "NOT OK"
            POS_1_I2_STATUS = "NOT OK"
            POS_2_I3_STATUS = "NOT OK"
            POS_3_I4_STATUS = "NOT OK"
            POS_4_I5_STATUS = "NOT OK"
            POS_5_E4_STATUS = "NOT OK"
            POS_6_E3_STATUS = "NOT OK"
            POS_7_E2_STATUS = "NOT OK"
            POS_8_E1_STATUS = "NOT OK"

            POS_2_I2_STATUS = "NOT OK"

            POS_4_I4_STATUS = "NOT OK"
            POS_4_I5_STATUS = "NOT OK"
            CAM_CAP_NUMBER_STATUS = "NOT OK"
            position = ""
            OVREALL_STATUS = "NOT OK"

            CAM_CAP_NUMBER_STATUS_POS1 = "NOT OK"
            CAM_CAP_NUMBER_STATUS_POS2 = "NOT OK"
            CAM_CAP_NUMBER_STATUS_POS3 = "NOT OK"
            CAM_CAP_NUMBER_STATUS_POS4 = "NOT OK"
            CAM_CAP_NUMBER_STATUS_POS5 = "NOT OK"
            CAM_CAP_NUMBER_STATUS_POS6 = "NOT OK"
            CAM_CAP_NUMBER_STATUS_POS7 = "NOT OK"
            CAM_CAP_NUMBER_STATUS_POS8 = "NOT OK"

            HLA_PostionCheck_pos1 = "NOT OK"
            HLA_PostionCheck_pos2 = "NOT OK"
            HLA_PostionCheck_pos3 = "NOT OK"
            HLA_PostionCheck_pos4 = "NOT OK"
            HLA_PostionCheck_pos5 = "NOT OK"
            HLA_PostionCheck_pos6 = "NOT OK"
            HLA_PostionCheck_pos7 = "NOT OK"
            HLA_PostionCheck_pos8 = "NOT OK"

            OVERALL_CAM_CAP_RESULT = "NOT OK"
            OVERALL_HLA_RESULT = "NOT OK"

            #4_Cylinder = 1 = I2, 2 = I3, 3 = I4, 4 = I5, 5 = E4, 6 = E3, 7 = E2, 8 = E1
            #3_Cylinder = 1 = I2, 2 = I3, 3 = I4, 4 = E3, 5 = E2, 6 = E1

            Engine_no, Engine_Type = "0301AAW00010N_CT12345678","4_Cylinder"
            if DEMO_RUN_FLAG is False:
                time.sleep(0.5)
                Engine_no, Engine_Type = getInferenceTrigger()
            if Engine_Type in ["3_Cylinder", "4_Cylinder"]:             
                try:                  
                    img_list = sorted([img for img in os.listdir(CAM_IMAGE) if img.endswith(".jpg")])
                    if len(img_list) == 0:
                        print("=====================image not found=====================")
                        time.sleep(0.1)
                        continue

                    try:
                        if "_" in Engine_no:
                            Engine_serialNumber = "_".join(Engine_no.split("_")[1:])
                        else:
                            Engine_serialNumber = Engine_no  # or "", depending on what you want

                        print("Final Engine_serialNumber:", Engine_serialNumber)

                    except Exception as e:
                        print("Exception in Engine_serialNumber",e) 
                    if Engine_Type == "4_Cylinder":
                        try:
                            if len(img_list) >= 8:
                                time.sleep(0.1)
                               # ENGINE_VARATION = Engine_no[0]+Engine_no[1]
                                ENGINE_VARATION = Engine_no[14]+Engine_no[15]
                                logger.debug(f"Cycle start for Engine_no: {Engine_no}, Engine_Type: {Engine_Type}")
                                for img_name in img_list:
                                    if ".jpg" not in img_name:
                                        continue

                                    if img_name == "POS_1.jpg":
                                        position = "1"
                                    elif img_name == "POS_2.jpg":
                                        position = "2"
                                    elif img_name == "POS_3.jpg":
                                        position = "3"
                                    elif img_name == "POS_4.jpg":
                                        position = "4"
                                    elif img_name == "POS_5.jpg":
                                        position = "5"
                                    elif img_name == "POS_6.jpg":
                                        position = "6"
                                    elif img_name == "POS_7.jpg":
                                        position = "7"
                                    elif img_name == "POS_8.jpg":
                                        position = "8"

                                    
                                    TodaysDate = datetime.datetime.now().strftime('%Y_%m_%d')
                                    if os.path.exists(os.path.join(SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                        os.mkdir(os.path.join(SAVED_FOLDER_PATH, TodaysDate))
                                    if os.path.exists(os.path.join(os.path.join(SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                        os.mkdir(os.path.join(os.path.join(SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                    image_folder_link1 = os.path.join(SAVED_FOLDER_PATH, TodaysDate)
                                    engine_folder_path=(image_folder_link1+"/"+Engine_no)+"/"

                                    time.sleep(0.1)
                                    im = cv2.imread(os.path.join(CAM_IMAGE,img_name))
                                    time.sleep(0.1)
                                    imreal = im.copy()
                                    time.sleep(0.1)
                                    original_image = im.copy()
                                                       
                                    original_image,OBJECT_LIST = maskRCNNObj.run_inference_new(im)
                                    #print("OBJECT_LIST is =================",OBJECT_LIST)
                                    logger.debug(f"For Engine_no is========== {Engine_no} process start")
                                  
                                    if OBJECT_LIST == [] or OBJECT_LIST is None:
                                        original_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                        if position == "1":  # Position I2
                                            POS_1_I2_STATUS = "NOT OK"
                                            CAM_I2_IMAGE_LINK = save_cam_image_for_mt_list(position="1", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_1_I2_STATUS", 
                                                                            image_suffix="Pos01", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "2":  # Position I3
                                            POS_2_I3_STATUS = "NOT OK"
                                            CAM_I3_IMAGE_LINK = save_cam_image_for_mt_list(position="2", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_2_I3_STATUS", 
                                                                            image_suffix="Pos02", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "3":  # Position I4
                                            POS_3_I4_STATUS = "NOT OK"
                                            CAM_I4_IMAGE_LINK = save_cam_image_for_mt_list(position="3", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_3_I4_STATUS", 
                                                                            image_suffix="Pos03", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "4":  # Position I5
                                            POS_4_I5_STATUS = "NOT OK"
                                            CAM_I5_IMAGE_LINK = save_cam_image_for_mt_list(position="4", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_4_I5_STATUS", 
                                                                            image_suffix="Pos04", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "5":  # Position E4
                                            POS_5_E4_STATUS = "NOT OK"
                                            CAM_E4_IMAGE_LINK = save_cam_image_for_mt_list(position="5", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_5_E4_STATUS", 
                                                                            image_suffix="Pos05", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "6":  # Position E3
                                            CAM_CAP_NUMBER_STATUS_POS6 = "NOT OK"
                                            CAM_E3_IMAGE_LINK = save_cam_image_for_mt_list(position="6", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_6_E3_STATUS", 
                                                                            image_suffix="Pos06", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "7":  # Position E2
                                            CAM_E2_IMAGE_LINK = save_cam_image_for_mt_list(position="7", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_7_E2_STATUS", 
                                                                            image_suffix="Pos07", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "8":  # Position E1
                                            CAM_E1_IMAGE_LINK = save_cam_image_for_mt_list(position="8", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_8_E1_STATUS", 
                                                                            image_suffix="Pos08", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)
                                
                                    else:                                                                                                                                                  
                                        #============================= Postion 2 ==================================#
                                        if position == "1":#=============== I2                                         
                                            logger.debug(f"ENGINE_VARATION is ============{ENGINE_VARATION}")
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos1,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS1,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image, 1)
                                            else:
                                                HLA_PostionCheck_pos1,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS1,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image, 2)
                           
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_I2_4C for sublist in OBJECT_LIST):
                                                print("n OK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False
                                            
                                            if CAM_CAP_NUMBER_STATUS_POS1 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS1 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos01_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos1.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)
                                            
                                            CAM_CAP_NUMBER_STATUS_POS1 = "OK"
                                            if HLA_PostionCheck_pos1 == "OK" and CAM_CAP_NUMBER_STATUS_POS1 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                              # putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_1_I2_STATUS = "OK"
                                            else:
                                                # pos = (50, 60)
                                                # putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_1_I2_STATUS = "NOT OK"
                                                
                                                if CAM_CAP_NUMBER_STATUS_POS1 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos1 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos)
                                            
                                            if POS_1_I2_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos01_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos01.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos01_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos01.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            original_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, original_image)
                                            CAM_I2_IMAGE_LINK = cam1_image_path
                                    
                                        #============================= Postion 3 =================================#
                                        if position == "2":#=============== I3  CAM4_                   
                                            
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos2,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS2,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,1)
                                            else:
                                                HLA_PostionCheck_pos2,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS2,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,2)
                                           
                                            if CAM_CAP_NUMBER_STATUS_POS2 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS2 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos02_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos2.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_I3_4C for sublist in OBJECT_LIST):    
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            CAM_CAP_NUMBER_STATUS_POS2 = "OK"
                                            if HLA_PostionCheck_pos2 == "OK" and CAM_CAP_NUMBER_STATUS_POS2 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_2_I3_STATUS = "OK"
                                            else:
                                                #pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_2_I3_STATUS = "NOT OK" 
                                                if CAM_CAP_NUMBER_STATUS_POS2 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos2 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos)

                                            if POS_2_I3_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos02_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos2.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos02_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos02.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_I3_IMAGE_LINK = cam1_image_path

                                            #============================= Postion 4 ==================================#
                                        if position == "3":#=============== I4
                                            
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos3,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS3,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,1)
                                            else:
                                                HLA_PostionCheck_pos3,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS3,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,2)
                                            
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_I4_4C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True

                                            if CAM_CAP_NUMBER_STATUS_POS3 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS3 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos03_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos3.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            CAM_CAP_NUMBER_STATUS_POS3 = "OK"
                                            if HLA_PostionCheck_pos3 == "OK" and CAM_CAP_NUMBER_STATUS_POS3 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_3_I4_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_3_I4_STATUS = "NOT OK" 
                                                if CAM_CAP_NUMBER_STATUS_POS3 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos3 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos)
                                            

                                            if POS_3_I4_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"
                                            
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos03_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos3.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos03_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos03.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_I4_IMAGE_LINK = cam1_image_path
                                        
                                        if position == "4":#=============== I5                                            
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos4,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS4,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image, 1)
                                            else:    
                                                HLA_PostionCheck_pos4,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS4,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image, 2)
                                            
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_I5_4C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False
                                            
                                            if CAM_CAP_NUMBER_STATUS_POS4 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS4 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos04_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos4.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            CAM_CAP_NUMBER_STATUS_POS4 = "OK"
                                            if HLA_PostionCheck_pos4 == "OK" and CAM_CAP_NUMBER_STATUS_POS4 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_4_I5_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_4_I5_STATUS = "NOT OK"

                                                if CAM_CAP_NUMBER_STATUS_POS4 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos4 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 
                                            
                                            if POS_4_I5_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"
                                            
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos04_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos4.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos04_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos04.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_I5_IMAGE_LINK = cam1_image_path
                                        
                                        if position == "5":#=============== E4                                            
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos5,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS5,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,1)
                                            else:
                                                HLA_PostionCheck_pos5,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS5,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,2)

                                            if CAM_CAP_NUMBER_STATUS_POS5 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS5 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos05_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos5.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)
                                            
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_E4_4C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True# False
                                            
                                            CAM_CAP_NUMBER_STATUS_POS5 = "OK"
                                            if HLA_PostionCheck_pos5 == "OK" and CAM_CAP_NUMBER_STATUS_POS5 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_5_E4_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_5_E4_STATUS = "NOT OK" 

                                                if CAM_CAP_NUMBER_STATUS_POS5 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos5 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 
                                            
                                            if POS_5_E4_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"
                                            
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos05_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos5.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)
                                            
                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos05_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos05.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_E4_IMAGE_LINK = cam1_image_path
                                        
                                        if position == "6":#=============== E3                                            
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos6,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS6,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,1)
                                            else:
                                                HLA_PostionCheck_pos6,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS6,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,2)

                                            if CAM_CAP_NUMBER_STATUS_POS6 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS6 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos06_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos6.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)
                                            
                                            
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_E3_4C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            CAM_CAP_NUMBER_STATUS_POS6 = "OK"
                                            if HLA_PostionCheck_pos6 == "OK" and CAM_CAP_NUMBER_STATUS_POS6 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                              #  putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_6_E3_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_6_E3_STATUS = "NOT OK" 

                                                if CAM_CAP_NUMBER_STATUS_POS6 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos6 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 
                                            
                                            if POS_6_E3_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos06_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos6.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos06_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos06.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_E3_IMAGE_LINK = cam1_image_path
                                        
                                        if position == "7": #=============== E2                                            
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos7,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS7,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,1)
                                            else:
                                                HLA_PostionCheck_pos7,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS7,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,2)

                                            if CAM_CAP_NUMBER_STATUS_POS7 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS7 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos07_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos7.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)
                                            
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_E2_4C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False
                                            CAM_CAP_NUMBER_STATUS_POS7 = "OK"
                                            if HLA_PostionCheck_pos7 == "OK" and CAM_CAP_NUMBER_STATUS_POS7 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_7_E2_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_7_E2_STATUS = "NOT OK"

                                                if CAM_CAP_NUMBER_STATUS_POS7 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos7 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 

                                            if POS_7_E2_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos07_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos7.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos07_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos07.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_E2_IMAGE_LINK = cam1_image_path

                                        
                                        if position == "8":#=============== E1                                            
                                            if ENGINE_VARATION in ["ZT","CT","CS"]:
                                                HLA_PostionCheck_pos8,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER_C50(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS8,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,1)
                                            else:
                                                HLA_PostionCheck_pos8,original_image = HLA_PositionCheck_WITH_ROD_4CYLINDER(position,OBJECT_LIST,original_image)
                                                CAM_CAP_NUMBER_STATUS_POS8,original_image = HLA_CAM_CAP_INSPECTION_4CYLINDER(position,OBJECT_LIST,original_image,2)

                                            if CAM_CAP_NUMBER_STATUS_POS8 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS8 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos08_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos8.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)
                                            
                                            
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_E1_4C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            CAM_CAP_NUMBER_STATUS_POS8 = "OK"
                                            if HLA_PostionCheck_pos8 == "OK" and CAM_CAP_NUMBER_STATUS_POS8 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_8_E1_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_8_E1_STATUS = "NOT OK"

                                                if CAM_CAP_NUMBER_STATUS_POS8 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos8 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 

                                            if POS_8_E1_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos08_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos8.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos08_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos08.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_E1_IMAGE_LINK = cam1_image_path


                                        
                                if POS_1_I2_STATUS == "OK" and POS_2_I3_STATUS == "OK"and POS_3_I4_STATUS == "OK" and POS_4_I5_STATUS == "OK" and POS_5_E4_STATUS == "OK" and POS_6_E3_STATUS == "OK" and POS_7_E2_STATUS == "OK" and POS_8_E1_STATUS == "OK":
                                    OVREALL_STATUS = "OK"

                                else:
                                    OVREALL_STATUS = "NOT OK"
                                
                                if CAM_CAP_NUMBER_STATUS_POS1 == "OK" and CAM_CAP_NUMBER_STATUS_POS2 == "OK"and CAM_CAP_NUMBER_STATUS_POS3 == "OK" and CAM_CAP_NUMBER_STATUS_POS4 == "OK" and CAM_CAP_NUMBER_STATUS_POS5 == "OK" and CAM_CAP_NUMBER_STATUS_POS6 == "OK" and CAM_CAP_NUMBER_STATUS_POS7 == "OK" and CAM_CAP_NUMBER_STATUS_POS8 == "OK":
                                    OVERALL_CAM_CAP_RESULT = "OK"
                                
                                if HLA_PostionCheck_pos1 == "OK" and HLA_PostionCheck_pos2 == "OK"and HLA_PostionCheck_pos3 == "OK" and HLA_PostionCheck_pos4 == "OK" and HLA_PostionCheck_pos5 == "OK" and HLA_PostionCheck_pos6 == "OK" and HLA_PostionCheck_pos7 == "OK" and HLA_PostionCheck_pos8 == "OK":
                                    OVERALL_HLA_RESULT = "OK"
                 
                                current_datetime = datetime.datetime.now()
                                processDateTime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")    
                                EngineTypeProcess = "4_Cylinder"                                                                   
                                IS_PROCESS_INF = insertDataIndotprocessing_table(Engine_no,EngineTypeProcess, OVREALL_STATUS, POS_1_I2_STATUS, POS_2_I3_STATUS, POS_3_I4_STATUS, POS_4_I5_STATUS, POS_5_E4_STATUS, POS_6_E3_STATUS, POS_7_E2_STATUS, POS_8_E1_STATUS,
                                                                                                 CAM_I2_IMAGE_LINK, CAM_I3_IMAGE_LINK,CAM_I4_IMAGE_LINK,CAM_I5_IMAGE_LINK,CAM_E4_IMAGE_LINK,CAM_E3_IMAGE_LINK,CAM_E2_IMAGE_LINK,CAM_E1_IMAGE_LINK)
                                IS_PROCESS_INF = "2"
                                
                                update_vision_overall_Status(OVREALL_STATUS)
                                pdfFIle_Name = start_pdf(OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT, Engine_no,engine_folder_path)    
                                insert_CamCap_HLA_Status_table_4CYLINDER(Engine_no,EngineTypeProcess, OVERALL_CAM_CAP_RESULT, CAM_CAP_NUMBER_STATUS_POS1, CAM_CAP_NUMBER_STATUS_POS2, CAM_CAP_NUMBER_STATUS_POS3, CAM_CAP_NUMBER_STATUS_POS4, CAM_CAP_NUMBER_STATUS_POS5, CAM_CAP_NUMBER_STATUS_POS6, CAM_CAP_NUMBER_STATUS_POS7, CAM_CAP_NUMBER_STATUS_POS8,
                                                                                                 OVERALL_HLA_RESULT, HLA_PostionCheck_pos1, HLA_PostionCheck_pos2,HLA_PostionCheck_pos3,HLA_PostionCheck_pos4,HLA_PostionCheck_pos5,HLA_PostionCheck_pos6,HLA_PostionCheck_pos7,HLA_PostionCheck_pos8,pdfFIle_Name)
                                
                               # pdfFIle_Name = start_pdf(OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT, Engine_no,engine_folder_path)
                                overAllStatus_update_ForPLC(Engine_no,OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT,IS_PROCESS_INF)
                                for i in img_list :
                                    if "POS_1.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_2.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_3.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_4.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    
                                    elif "POS_5.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_6.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_7.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_8.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                        print("========Remove IMG, PROCESS DONE==========") 

                        except Exception as e:
                            logger.debug(f"except Exception imag inf for {position} and error {e}")
                            print("except Exception INF IMAGE",e)


                    if Engine_Type == "3_Cylinder":
                        try:
                            if len(img_list) >= 6:
                                time.sleep(0.3)
                                logger.debug(f"Cycle start for Engine_no: {Engine_no}, Engine_Type: {Engine_Type}")
                                for img_name in img_list:
                                    if ".jpg" not in img_name:
                                        continue

                                    if img_name == "POS_1.jpg":
                                        position = "1"
                                    elif img_name == "POS_2.jpg":
                                        position = "2"
                                    elif img_name == "POS_3.jpg":
                                        position = "3"
                                    elif img_name == "POS_4.jpg":
                                        position = "4"
                                    elif img_name == "POS_5.jpg":
                                        position = "5"
                                    elif img_name == "POS_6.jpg":
                                        position = "6"
                                 
                                    
                                    TodaysDate = datetime.datetime.now().strftime('%Y_%m_%d')
                                    if os.path.exists(os.path.join(SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                        os.mkdir(os.path.join(SAVED_FOLDER_PATH, TodaysDate))
                                    if os.path.exists(os.path.join(os.path.join(SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                        os.mkdir(os.path.join(os.path.join(SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                    image_folder_link1 = os.path.join(SAVED_FOLDER_PATH, TodaysDate)
                                    engine_folder_path=(image_folder_link1+"/"+Engine_no)+"/"


                                    time.sleep(0.1)
                                    im = cv2.imread(os.path.join(CAM_IMAGE,img_name))
                                    time.sleep(0.1)
                                    imreal = im.copy()
                                    time.sleep(0.1)
                                    original_image = im.copy()
                                    
                                                 
                                    original_image,OBJECT_LIST = maskRCNNObj.run_inference_new(im)        

                                    logger.debug(f"For Engine_no is========== {Engine_no} process start")
                                    if OBJECT_LIST == [] or OBJECT_LIST is None:
                                        original_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                        if position == "1":  # Position I2
                                            CAM_I1_IMAGE_LINK = save_cam_image_for_mt_list(position="1", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_1_I1_STATUS", 
                                                                            image_suffix="Pos01", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "2":  # Position I3
                                            CAM_I2_IMAGE_LINK = save_cam_image_for_mt_list(position="2", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_2_I2_STATUS", 
                                                                            image_suffix="Pos02", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "3":  # Position I4
                                            CAM_I3_IMAGE_LINK = save_cam_image_for_mt_list(position="3", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_3_I3_STATUS", 
                                                                            image_suffix="Pos03", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "4":  # Position I5
                                            CAM_E3_IMAGE_LINK = save_cam_image_for_mt_list(position="4", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_6_E3_STATUS", 
                                                                            image_suffix="Pos04", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "5":  # Position E4
                                            CAM_E2_IMAGE_LINK = save_cam_image_for_mt_list(position="5", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_7_E2_STATUS", 
                                                                            image_suffix="Pos05", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)

                                        elif position == "6":  # Position E3
                                            CAM_E1_IMAGE_LINK = save_cam_image_for_mt_list(position="6", 
                                                                            status_text="CamCap is NotOK", 
                                                                            position_status_var="POS_8_E1_STATUS", 
                                                                            image_suffix="Pos06", 
                                                                            engine_folder_path=engine_folder_path, 
                                                                            original_image=original_image, 
                                                                            imreal=imreal)
                                        
                                    else:                                                                                                             
                                        #============================= Postion 2 ==================================#
                                        if position == "1":#=============== I2
                                            CAM_CAP_NUMBER_STATUS_POS1,original_image = HLA_CAM_CAP_INSPECTION_3CYLINDER(position,OBJECT_LIST,original_image)
                                            
                                            if CAM_CAP_NUMBER_STATUS_POS1 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS1 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos01_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos1.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            CAM_CAP_NUMBER_STATUS_POS1 = "OK"
                                            HLA_PostionCheck_pos1,original_image = HLA_PositionCheck_WITH_ROD_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_I1_3C for sublist in OBJECT_LIST):
                                                print("NOTOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            if HLA_PostionCheck_pos1 == "OK" and CAM_CAP_NUMBER_STATUS_POS1 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_1_I1_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_1_I1_STATUS = "NOT OK"

                                                if CAM_CAP_NUMBER_STATUS_POS1 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos1 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 
                                            if POS_1_I1_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos01_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos1.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos01_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos01.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_I1_IMAGE_LINK = cam1_image_path
                                    

                                        #============================= Postion 3 =================================#
                                        if position == "2":
                                            CAM_CAP_NUMBER_STATUS_POS2,original_image = HLA_CAM_CAP_INSPECTION_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if CAM_CAP_NUMBER_STATUS_POS2 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS2 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos02_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos2.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            CAM_CAP_NUMBER_STATUS_POS2 = "OK"
                                            HLA_PostionCheck_pos2,original_image = HLA_PositionCheck_WITH_ROD_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_I2_3C for sublist in OBJECT_LIST):
                                                print("NOTOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            if HLA_PostionCheck_pos2 == "OK" and CAM_CAP_NUMBER_STATUS_POS2 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_2_I2_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_2_I2_STATUS = "NOT OK" 

                                                if CAM_CAP_NUMBER_STATUS_POS2 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos2 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 
                                            
                                            if POS_2_I2_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos02_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos2.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)


                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos02_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos02.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_I2_IMAGE_LINK = cam1_image_path

                                            #============================= Postion 4 ==================================#
                                        if position == "3":
                                            CAM_CAP_NUMBER_STATUS_POS3,original_image = HLA_CAM_CAP_INSPECTION_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if CAM_CAP_NUMBER_STATUS_POS3 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS3 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos03_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos3.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            
                                            CAM_CAP_NUMBER_STATUS_POS3 = "OK"
                                            HLA_PostionCheck_pos3,original_image = HLA_PositionCheck_WITH_ROD_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_I3_3C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            if HLA_PostionCheck_pos3 == "OK" and CAM_CAP_NUMBER_STATUS_POS3 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_3_I3_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_3_I3_STATUS = "NOT OK" 

                                                if CAM_CAP_NUMBER_STATUS_POS3 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos3 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 

                                            if POS_3_I3_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos03_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos3.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos03_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos03.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_I3_IMAGE_LINK = cam1_image_path
                                        
                                        if position == "4":
                                            CAM_CAP_NUMBER_STATUS_POS5,original_image = HLA_CAM_CAP_INSPECTION_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if CAM_CAP_NUMBER_STATUS_POS5 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS5 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos04_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos4.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)
                                            
                                            CAM_CAP_NUMBER_STATUS_POS5 = "OK"
                                            HLA_PostionCheck_pos5,original_image = HLA_PositionCheck_WITH_ROD_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_E3_3C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            if HLA_PostionCheck_pos5 == "OK" and CAM_CAP_NUMBER_STATUS_POS5 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_6_E3_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_6_E3_STATUS = "NOT OK" 

                                                if CAM_CAP_NUMBER_STATUS_POS5 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos5 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 

                                            if POS_6_E3_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos04_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos4.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos04_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos04.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_E3_IMAGE_LINK = cam1_image_path
                                        
                                        if position == "5":
                                            CAM_CAP_NUMBER_STATUS_POS6,original_image = HLA_CAM_CAP_INSPECTION_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if CAM_CAP_NUMBER_STATUS_POS6 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS6 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos05_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos5.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            CAM_CAP_NUMBER_STATUS_POS6 = "OK"
                                            HLA_PostionCheck_pos6,original_image = HLA_PositionCheck_WITH_ROD_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_E2_3C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            if HLA_PostionCheck_pos6 == "OK" and CAM_CAP_NUMBER_STATUS_POS6 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_7_E2_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_7_E2_STATUS = "NOT OK" 

                                                if CAM_CAP_NUMBER_STATUS_POS6 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos6 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos) 
                                            
                                            if POS_7_E2_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos05_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos5.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok)
                                            
                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos05_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos05.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_E2_IMAGE_LINK = cam1_image_path
                                        
                                        if position == "6":
                                            CAM_CAP_NUMBER_STATUS_POS7,original_image = HLA_CAM_CAP_INSPECTION_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if CAM_CAP_NUMBER_STATUS_POS7 == "NOT_OK" or CAM_CAP_NUMBER_STATUS_POS7 == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_cap_not_ok = os.path.join(NOT_OK_CAP_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_cap_not_ok=(image_folder_link1_cap_not_ok+"/"+Engine_no)+"/"
                                                
                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok, formatted_dateime+"_Pos06_real.jpg")
                                                cam1_image_path_cap_not_ok = os.path.join(engine_folder_path_cap_not_ok,formatted_dateime+"_Pos6.jpg")
                                                cv2.imwrite(realcam1_image_path_cap_not_ok, imreal)
                                                original_image_cap_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_cap_not_ok, original_image_cap_not_ok)

                                            CAM_CAP_NUMBER_STATUS_POS7 = "OK"
                                            HLA_PostionCheck_pos7 ,original_image = HLA_PositionCheck_WITH_ROD_3CYLINDER(position,OBJECT_LIST,original_image)
                                            if any(class_name in sublist[0] for class_name in CLASS_NAME_OK_LIST_E1_3C for sublist in OBJECT_LIST):
                                                print("nOK")
                                                class_check_cam1 = False
                                            else:
                                                class_check_cam1 = True#False

                                            if HLA_PostionCheck_pos7 == "OK" and CAM_CAP_NUMBER_STATUS_POS7 == "OK" and class_check_cam1 is True:
                                                pos = (50, 60)
                                                #putText_notok_ok("CamCap is OK",original_image, pos)
                                                POS_8_E1_STATUS = "OK"
                                            else:
                                                pos = (50, 60)
                                               # putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                POS_8_E1_STATUS = "NOT OK" 

                                                if CAM_CAP_NUMBER_STATUS_POS7 == "NOT OK" or class_check_cam1 is False :
                                                    pos = (50, 60)
                                                    putText_notok_ok("CamCap is NotOK",original_image, pos)
                                                if HLA_PostionCheck_pos7 == "NOT OK":
                                                    pos = (50, 60)
                                                    putText_notok_ok("HLA is NotOK",original_image, pos)

                                            
                                            if POS_8_E1_STATUS == "NOT OK":
                                                if os.path.exists(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)) is False: 
                                                    os.mkdir(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate))
                                                if os.path.exists(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no)) is False:
                                                    os.mkdir(os.path.join(os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate),Engine_no))
                                                image_folder_link1_not_ok = os.path.join(NOT_OK_SAVED_FOLDER_PATH, TodaysDate)
                                                engine_folder_path_not_ok=(image_folder_link1_not_ok+"/"+Engine_no)+"/"

                                                current_datetime = datetime.datetime.now()
                                                formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                                # Generate image paths including the folder structure
                                                realcam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok, formatted_dateime+"_Pos06_real.jpg")
                                                cam1_image_path_not_ok = os.path.join(engine_folder_path_not_ok,formatted_dateime+"_Pos6.jpg")
                                                cv2.imwrite(realcam1_image_path_not_ok, imreal)
                                                original_image_not_ok = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                cv2.imwrite(cam1_image_path_not_ok, original_image_not_ok) 

                                            current_datetime = datetime.datetime.now()
                                            formatted_dateime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                                            # Generate image paths including the folder structure
                                            realcam1_image_path = os.path.join(engine_folder_path, formatted_dateime+"_Pos06_real.jpg")
                                            cam1_image_path = os.path.join(engine_folder_path,formatted_dateime+"_Pos06.jpg")
                                            cv2.imwrite(realcam1_image_path, imreal)
                                            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            cv2.imwrite(cam1_image_path, rotated_image)
                                            CAM_E1_IMAGE_LINK = cam1_image_path
                                        
                                        
                                        
                                if POS_1_I1_STATUS == "OK" and POS_2_I2_STATUS == "OK" and POS_3_I3_STATUS == "OK" and POS_6_E3_STATUS == "OK" and POS_7_E2_STATUS == "OK" and POS_8_E1_STATUS == "OK":
                                    OVREALL_STATUS = "OK"

                                if CAM_CAP_NUMBER_STATUS_POS1 == "OK" and CAM_CAP_NUMBER_STATUS_POS2 == "OK" and CAM_CAP_NUMBER_STATUS_POS3 == "OK" and CAM_CAP_NUMBER_STATUS_POS5 == "OK" and CAM_CAP_NUMBER_STATUS_POS6 == "OK" and CAM_CAP_NUMBER_STATUS_POS7 == "OK":
                                    OVERALL_CAM_CAP_RESULT = "OK"
                                
                                if HLA_PostionCheck_pos1 == "OK" and HLA_PostionCheck_pos2 == "OK" and HLA_PostionCheck_pos3 == "OK" and HLA_PostionCheck_pos5 == "OK" and HLA_PostionCheck_pos6 == "OK" and HLA_PostionCheck_pos7 == "OK":
                                    OVERALL_HLA_RESULT = "OK"

                                else:
                                    OVREALL_STATUS = "NOT OK"


                                current_datetime = datetime.datetime.now()
                                processDateTime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")    
                                EngineTypeProcess = "3_Cylinder"   
                                POS_4_I4_STATUS = ''
                                POS_5_E4_STATUS = '' 
                                CAM_I4_IMAGE_LINK = ''
                                CAM_E4_IMAGE_LINK = ''
                                CAM_CAP_NUMBER_STATUS_POS4= ''
                                CAM_CAP_NUMBER_STATUS_POS8= ''   
                                HLA_PostionCheck_pos4= '' 
                                HLA_PostionCheck_pos8= ''                                                
                                IS_PROCESS_INF = insertDataIndotprocessing_table(Engine_no,EngineTypeProcess, OVREALL_STATUS, POS_1_I1_STATUS, POS_2_I2_STATUS, POS_3_I3_STATUS, POS_4_I4_STATUS, POS_5_E4_STATUS, POS_6_E3_STATUS, POS_7_E2_STATUS, POS_8_E1_STATUS,
                                                                                                 CAM_I1_IMAGE_LINK, CAM_I2_IMAGE_LINK,CAM_I3_IMAGE_LINK,CAM_I4_IMAGE_LINK,CAM_E4_IMAGE_LINK,CAM_E3_IMAGE_LINK,CAM_E2_IMAGE_LINK,CAM_E1_IMAGE_LINK)
                                
                                pdfFIle_Name = start_pdf(OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT, Engine_no,engine_folder_path)
                                insert_CamCap_HLA_Status_table_3CYLINDER(Engine_no,EngineTypeProcess, OVERALL_CAM_CAP_RESULT, CAM_CAP_NUMBER_STATUS_POS1, CAM_CAP_NUMBER_STATUS_POS2, CAM_CAP_NUMBER_STATUS_POS3, CAM_CAP_NUMBER_STATUS_POS5, CAM_CAP_NUMBER_STATUS_POS6, CAM_CAP_NUMBER_STATUS_POS7,
                                                                                                 OVERALL_HLA_RESULT, HLA_PostionCheck_pos1, HLA_PostionCheck_pos2,HLA_PostionCheck_pos3,HLA_PostionCheck_pos5,HLA_PostionCheck_pos6,HLA_PostionCheck_pos7,pdfFIle_Name)
                                IS_PROCESS_INF = "2"   
                                overAllStatus_update_ForPLC(Engine_no,OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT,IS_PROCESS_INF)
                                update_vision_overall_Status(OVREALL_STATUS)
                                #pdfFIle_Name = start_pdf(OVREALL_STATUS,OVERALL_CAM_CAP_RESULT,OVERALL_HLA_RESULT, Engine_no,engine_folder_path)
                                for i in img_list :
                                    if "POS_1.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_2.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_3.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_4.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    
                                    elif "POS_5.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_6.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_7.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))
                                    elif "POS_8.jpg" in i:
                                        os.remove(os.path.join(CAM_IMAGE,i))

                                        print("========Remove IMG, PROCESS DONE==========") 

                        except Exception as e:
                            logger.debug(f"except Exception imag inf for {position} and error {e}")
                            print("except Exception INF IMAGE",e)            
 
                except Exception as e:
                    print(e)
                    logger.debug(f"except Exception check_for_defect for {position} and error {e}")
                    logger.debug(f"Error in check_for_defect(): {e}")



    except Exception as e:
        print(e)
        print(traceback.format_exc())
        logger.debug(f"Error in run(): {e}")


def initializeLogger():
    global logger
    import logging
    from logging.handlers import RotatingFileHandler
    import os
    import traceback
    try:
        ''' Initializing Logger '''
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # Define the log file format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Ensure the base directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        log_file = os.path.join(base_dir, os.path.splitext(os.path.basename(__file__))[0] + ".log")

        # Rotating file handler (size-based rotation)
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)  # 10 MB per file
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Adding the handler to the logger
        logger.addHandler(file_handler)

        logger.debug("Algorithm Module Initialized")
    except Exception as e:
        print(f"initializeLogger() Exception is {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    initializeLogger()
    maskRCNNObj = MaskRCNN_Mahindra()
    run(maskRCNNObj)
