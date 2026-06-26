import json
with open('Notebook/BOC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class AdvancedFeatureEngineer' in source:
            print(source)