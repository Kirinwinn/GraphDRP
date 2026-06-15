import pandas as pd
valid_smiles = ['C'] * 128
predictions = [0.5] * 129
try:
    df = pd.DataFrame({"sm": valid_smiles, "pred": predictions})
except Exception as e:
    print(e)
