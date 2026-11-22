from pathlib import Path
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
a=AppTest.from_file(str(root/'dashboard/app.py'),default_timeout=90).run()
assert not a.exception, [e.message for e in a.exception]
a.selectbox[0].select('Accuracy (%)').run()
assert not a.exception
print('Dashboard: render e interacción correctos.')
