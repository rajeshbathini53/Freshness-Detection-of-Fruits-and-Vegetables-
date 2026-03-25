import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk,ImageOps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import cv2

train_features=[]
train_labels=[]
train_brightness=[]
train_paths=[]
test_features=[]
test_paths=[]
model=None
label_encoder=LabelEncoder()
preprocess_applied=False

def log(msg):
    output_box.insert(tk.END,msg+"\n")
    output_box.see(tk.END)
    root.update_idletasks()

def safe_open_image(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        return None

def resize_image(img,size):
    try:
        return img.resize(size)
    except:
        return img

def extract_image_features(img):
    arr=np.array(img)
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    brightness=np.mean(gray)
    color_var=np.var(arr)
    sharpness=cv2.Laplacian(gray,cv2.CV_64F).var()
    return [brightness,color_var,sharpness],brightness

def extract_labeled_features(base_folder):
    features=[]
    labels=[]
    brightness=[]
    paths=[]
    for label in os.listdir(base_folder):
        class_path=os.path.join(base_folder,label)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg','.jpeg','.png')):
                fpath=os.path.join(class_path,file)
                img=safe_open_image(fpath)
                if img is None:
                    continue
                if preprocess_applied:
                    img=ImageOps.exif_transpose(img)
                img=resize_image(img,(224,224))
                feat,bright=extract_image_features(img)
                features.append(feat)
                brightness.append(bright)
                labels.append(label.lower())
                paths.append(fpath)
    return np.array(features),np.array(labels),np.array(brightness),np.array(paths)

def extract_features(folder):
    feats=[]
    paths=[]
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            fpath=os.path.join(folder,file)
            img=safe_open_image(fpath)
            if img is None:
                continue
            if preprocess_applied:
                img=ImageOps.exif_transpose(img)
            img=resize_image(img,(224,224))
            feat,_=extract_image_features(img)
            feats.append(feat)
            paths.append(fpath)
    return np.array(feats),np.array(paths)

def load_train():
    folder=filedialog.askdirectory()
    if folder:
        global train_features,train_labels,train_brightness,train_paths
        train_features,train_labels,train_brightness,train_paths=extract_labeled_features(folder)
        log("✔ TRAIN DATASET LOADED SUCCESSFULLY")
        log("• Total Training Images : "+str(len(train_labels)))
        if len(train_labels)>0:
            u,c=np.unique(train_labels,return_counts=True)
            for a,b in zip(u,c):
                log("• "+a.upper()+" : "+str(b))

def load_test():
    folder=filedialog.askdirectory()
    if folder:
        global test_features,test_paths
        test_features,test_paths=extract_features(folder)
        log("✔ TEST DATASET LOADED SUCCESSFULLY")
        log("• Total Test Images : "+str(len(test_features)))

def apply_preprocess():
    global preprocess_applied
    preprocess_applied=True
    log("✔ PREPROCESSING COMPLETED SUCCESSFULLY")

def train_model():
    global model
    if len(train_features)==0:
        log("❌ Load training data first")
        return
    label_encoder.fit(train_labels)
    y=label_encoder.transform(train_labels)
    model=RandomForestClassifier(n_estimators=300,random_state=42)
    model.fit(train_features,y)
    acc=np.mean(model.predict(train_features)==y)
    log("✔ MODEL TRAINED SUCCESSFULLY")
    log("• Algorithm : Random Forest Classifier")
    log("• Training Accuracy : "+str(round(acc*100,2))+" %")

def eda_brightness():
    if len(train_features)==0:
        log("❌ Load training data first")
        return
    plt.figure(figsize=(7,4))
    sns.histplot(x=train_brightness,hue=train_labels,bins=25,kde=True)
    plt.title("Brightness Distribution")
    plt.show()
    log("✔ Brightness Plot Displayed")

def eda_pca():
    if len(train_features)==0:
        log("❌ Load training data first")
        return
    pca=PCA(n_components=2)
    reduced=pca.fit_transform(train_features)
    plt.figure(figsize=(6,5))
    for lbl in np.unique(train_labels):
        plt.scatter(reduced[train_labels==lbl,0],reduced[train_labels==lbl,1],label=lbl)
    plt.legend()
    plt.title("PCA Distribution")
    plt.show()
    log("✔ PCA Plot Displayed")

def eda_class_count():
    if len(train_features)==0:
        log("❌ Load training data first")
        return
    labels,count=np.unique(train_labels,return_counts=True)
    plt.figure(figsize=(5,4))
    sns.barplot(x=labels,y=count)
    plt.title("Class Count")
    plt.show()
    log("✔ Class Count Plot Displayed")

def predict_image():
    if model is None:
        log("❌ Train model first")
        return
    img_path=filedialog.askopenfilename()
    if not img_path:
        return
    img=safe_open_image(img_path)
    if img is None:
        return
    if preprocess_applied:
        img=ImageOps.exif_transpose(img)
    img=resize_image(img,(224,224))
    feat,_=extract_image_features(img)
    pred=model.predict([feat])
    label=label_encoder.inverse_transform(pred)[0]

    for w in image_frame.winfo_children():
        w.destroy()

    disp=resize_image(Image.open(img_path),(220,220))
    imgtk=ImageTk.PhotoImage(disp)
    lbl=tk.Label(image_frame,image=imgtk)
    lbl.image=imgtk
    lbl.pack()

    result_label.config(text="PREDICTED : "+label.upper())
    log("✔ PREDICTION COMPLETED")
    log("• Predicted Freshness : "+label.upper())

root=tk.Tk()
root.title("Freshness Detection of Fruits and Vegetables")
root.geometry("1200x850")

title=tk.Label(root,text="Freshness Detection of Fruits and Vegetables",font=("Arial",18,"bold"))
title.pack(pady=6)

main_frame=tk.Frame(root)
main_frame.pack(fill="both",expand=True)

left_frame=tk.Frame(main_frame,width=240)
left_frame.pack(side="left",fill="y")
left_frame.pack_propagate(False)

right_frame=tk.Frame(main_frame)
right_frame.pack(side="left",fill="both",expand=True)

btn_color="lightgreen"
btn_opts={"width":24,"height":2,"bg":btn_color}

tk.Button(left_frame,text="Load Train Folder",command=load_train,**btn_opts).pack(pady=2,padx=(10,0))
tk.Button(left_frame,text="Load Test Folder",command=load_test,**btn_opts).pack(pady=2,padx=(10,0))
tk.Button(left_frame,text="Preprocess",command=apply_preprocess,**btn_opts).pack(pady=2,padx=(10,0))
tk.Button(left_frame,text="Train Model",command=train_model,**btn_opts).pack(pady=2,padx=(10,0))
tk.Button(left_frame,text="EDA Brightness Plot",command=eda_brightness,**btn_opts).pack(pady=2,padx=(10,0))
tk.Button(left_frame,text="EDA PCA Plot",command=eda_pca,**btn_opts).pack(pady=2,padx=(10,0))
tk.Button(left_frame,text="EDA Class Count",command=eda_class_count,**btn_opts).pack(pady=2,padx=(10,0))
tk.Button(left_frame,text="Predict Image",command=predict_image,**btn_opts).pack(pady=2,padx=(10,0))

output_box=tk.Text(right_frame,height=18,width=92)
output_box.pack(fill="both",expand=True,padx=0,pady=2)

result_label=tk.Label(right_frame,text="",font=("Arial",14,"bold"))
result_label.pack(pady=4)

image_frame=tk.Frame(right_frame)
image_frame.pack(pady=6)

root.mainloop()
