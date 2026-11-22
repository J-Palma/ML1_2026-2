"""Execute narrative notebooks in-process, capturing real rich display outputs."""
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1];shell=InteractiveShell.instance()
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
 shell.reset(new_session=False)
 n=nbformat.read(p,as_version=4);count=0
 for index,cell in enumerate(n.cells):
  if cell.cell_type!='code':continue
  count+=1
  title=n.cells[index-1].source.splitlines()[0].lstrip('# ') if index else 'Inicio'
  print(f'{p.name} · celda {count}: {title}',flush=True)
  with capture_output() as captured:result=shell.run_cell(cell.source,store_history=False)
  outputs=[]
  if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
  if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
  for out in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
  if result.error_in_exec or result.error_before_exec:raise RuntimeError(f'{p}: {result.error_in_exec or result.error_before_exec}')
  cell.outputs=outputs;cell.execution_count=count
  nbformat.write(n,p)
 nbformat.write(n,p);print('Ejecutado',p.name,flush=True)
