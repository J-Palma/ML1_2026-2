from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import workflow
for stage in ['prepare_data','extract','baseline','advanced','feature_engineering','evaluate']:
 print('\nETAPA:',stage,flush=True)
 getattr(workflow,stage)()
print('Pipeline final completado.',flush=True)
