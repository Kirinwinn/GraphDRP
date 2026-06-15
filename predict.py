import os
import csv
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from models.ginconv import GINConvNet
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from utils import TestbedDataset
from preprocess import smile_to_graph
from tqdm import tqdm

# ================= Configuration ====================
SMILES_CSV_FILE = "../XXX.csv"              # Path to the CSV file containing molecular SMILES
TARGET_CELL_ID = "1298134"                  # Cosmic ID for HeLa cell
MODEL_WEIGHT_PATH = "weights/GINConvNet/model_GINConvNet_GDSC.model" # Trained weights
MODEL_CHOICE = 0                            # 0: GINConvNet
DEVICE_NAME = "cuda:0"                      # Change to "cpu" if error occurs
OUTPUT_FILE = "hela_predictions.csv"        # Output file name
# =============================================

def load_target_cell_feature(cell_id_target):
    """
    Extract feature vector of the cell line (length 735 0/1 vector)
    """
    f = open("data/PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    
    mut_dict = {}
    cell_feature_dict = {}
    
    # 1. Build mutation dictionary
    for item in reader:
        mut = item[5]
        if mut not in mut_dict:
            mut_dict[mut] = len(mut_dict)
            
    # 2. Fill feature vector
    f.seek(0)
    next(reader)
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
    print(f'Predicting for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for data in tqdm(loader, desc="Predicting batches"):
            data = data.to(device)
            output, _ = model(data)
            total_preds.extend(output.cpu().numpy().flatten())
    return total_preds

if __name__ == "__main__":
    device = torch.device(DEVICE_NAME if torch.cuda.is_available() else "cpu")
    
    print(f"1. Extracting features for HeLa cell line (Cosmic ID: {TARGET_CELL_ID})...")
    cell_feature = load_target_cell_feature(TARGET_CELL_ID)
    if cell_feature is None:
        raise ValueError(f"Features for cell line {TARGET_CELL_ID} not found!")

    print(f"2. Extracting molecular graph features and assembling test set ({SMILES_CSV_FILE})...")
    df = pd.read_csv(SMILES_CSV_FILE)
    smiles_list = df['SMILES'].tolist()
    
    smile_graph = {}
    xd_test, xc_test = [], []
    valid_smiles = []
    
    for s in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            g = smile_to_graph(s)
            smile_graph[s] = g
            xd_test.append(s)
            xc_test.append(cell_feature)
            valid_smiles.append(s)
        except Exception as e:
            print(f"Failed to parse SMILES, skipping: {s} -> Error: {e}")

    xd_test = np.asarray(xd_test)
    xc_test = np.asarray(xc_test)
    fake_y = np.zeros(len(xd_test))

    test_data = TestbedDataset(root='data', dataset='my_infer', xd=xd_test, xt=xc_test, y=fake_y, smile_graph=smile_graph)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    print(f"3. Loading model and parameters ({MODEL_WEIGHT_PATH})...")
    model_classes = [GINConvNet, GATNet, GAT_GCN, GCNNet]
    model = model_classes[MODEL_CHOICE]().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device, weights_only=False))

    print("4. Starting inference...")
    predictions = predicting(model, device, test_loader)
    
    # === Inverse operation to restore true concentration units ===
    predictions = np.array(predictions)
    eps = 1e-7
    pred_clipped = np.clip(predictions, eps, 1.0 - eps)
    
    ln_ic50 = -10.0 * np.log((1.0 - pred_clipped) / pred_clipped)
    ic50_uM = np.exp(ln_ic50)
    
    out_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'Pred_Normalized_Score': predictions,
        'Pred_ln_IC50': ln_ic50,
        'Pred_IC50_uM': ic50_uM
    })
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Prediction completed!!! Results saved in GraphDRP/{OUTPUT_FILE}")
