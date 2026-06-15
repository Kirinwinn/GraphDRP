import os
import csv
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from tqdm import tqdm

from models.ginconv import GINConvNet
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from utils import TestbedDataset
from preprocess import smile_to_graph

INPUT_FOLDER = "/home/cenking/VsCode/BatchAll_Origin_F_All"               
OUTPUT_FOLDER = "/home/cenking/VsCode/GraphDRP/output"   

TARGET_CELL_ID = "1298134"                                
MODEL_WEIGHT_PATH = "/home/cenking/VsCode/GraphDRP/weights/GINConvNet/model_GINConvNet_GDSC.model"
MODEL_CHOICE = 0                                          
DEVICE_NAME = "cuda:0"                                    

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_target_cell_feature(cell_id_target):
    pancancer_feature_csv = "/home/cenking/VsCode/GraphDRP/data/PANCANCER_Genetic_feature.csv"
    with open(pancancer_feature_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        mut_dict = {}
        cell_feature_dict = {}
        for item in reader:
            mut = item[5]
            if mut not in mut_dict:
                mut_dict[mut] = len(mut_dict)
        f.seek(0)
        next(reader) # skip headers
        for item in reader:
            cell_id = item[1]
            mut = item[5]
            is_mutated = int(item[6])
            if cell_id not in cell_feature_dict:
                cell_feature_dict[cell_id] = np.zeros(len(mut_dict))
            col = mut_dict[mut]
            cell_feature_dict[cell_id][col] = is_mutated
        return cell_feature_dict.get(cell_id_target, None)

def predicting(model, device, loader):
    model.eval()
    total_preds = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Predicting", leave=False):
            data = data.to(device)
            output, _ = model(data)
            total_preds.extend(output.cpu().numpy().flatten())
    return total_preds

def run_batch_prediction():
    device = torch.device(DEVICE_NAME if torch.cuda.is_available() else "cpu")
    print(f"Loading features for cell line (Cosmic ID: {TARGET_CELL_ID})...")
    cell_feature = load_target_cell_feature(TARGET_CELL_ID)
    if cell_feature is None:
        raise ValueError(f"Features for cell line {TARGET_CELL_ID} not found. Please verify Cosmic ID.")

    print(f"Loading pre-trained model: {MODEL_WEIGHT_PATH}...")
    model_classes = [GINConvNet, GATNet, GAT_GCN, GCNNet]
    model = model_classes[MODEL_CHOICE]().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device, weights_only=False))
    
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} target file(s) in {INPUT_FOLDER}")
    
    for filename in tqdm(csv_files, desc="Batch File Progress"):
        input_csv_path = os.path.join(INPUT_FOLDER, filename)
        output_csv_path = os.path.join(OUTPUT_FOLDER, filename.replace(".csv", "_evaluated.csv"))
        df = pd.read_csv(input_csv_path)
        if 'SMILES' not in df.columns:
            continue
            
        smiles_list = df['SMILES'].tolist()[0:10] # test small bit
        smile_graph = {}
        xd_test, xc_test = [], []
        valid_smiles = []
        for s in smiles_list:
            try:
                g = smile_to_graph(s)
                smile_graph[s] = g
                xd_test.append(s)
                xc_test.append(cell_feature)
                valid_smiles.append(s)
            except Exception:
                pass
                
        if not valid_smiles:
            continue
            
        xd_test = np.asarray(xd_test)
        xc_test = np.asarray(xc_test)
        fake_y = np.zeros(len(xd_test))
        
        dataset_name = f'infer_{os.path.splitext(filename)[0]}'
        t_data = TestbedDataset(root='/home/cenking/VsCode/GraphDRP/data', dataset=dataset_name, xd=xd_test, xt=xc_test, y=fake_y, smile_graph=smile_graph)
        test_loader = DataLoader(t_data, batch_size=128, shuffle=False)
        
        predictions = predicting(model, device, test_loader)
        print("Success predicting batch!")
        break # Test one and exit
        
run_batch_prediction()
