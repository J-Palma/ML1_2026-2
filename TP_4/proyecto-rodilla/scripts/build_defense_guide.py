"""Render the reviewed oral-defense study guide; does not retrain or change results."""
from pathlib import Path
import hashlib
import html
import json
import re
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, KeepTogether, Image as ReportImage)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

R=Path(__file__).resolve().parents[1]
source=R/'presentation/guia_defensa.md'
text=source.read_text(encoding='utf-8-sig')
metrics=json.loads((R/'reports/modelos_avanzados_metrics.json').read_text(encoding='utf-8'))
if metrics['protocol'] not in text or metrics['selected_model'] not in text:
    raise ValueError('La guía necesita revisión para el modelo vigente.')
questions=re.findall(r'^\*\*(\d+)\. ',text,re.M)
question_count=len(questions)
assert list(map(int,questions))==list(range(1,question_count+1)), 'Preguntas numeradas sin saltos.'
assert question_count >= 100
assert len(re.findall(r'^### Diapositiva ',text,re.M))==14
for name,file in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(R/'presentation/assets/fonts'/file)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB',italic='DV',boldItalic='DVB')
NAVY=colors.HexColor('#102A43');TEAL=colors.HexColor('#007C83');GRAY=colors.HexColor('#52606D')
styles={
 'title':ParagraphStyle('title',fontName='DVB',fontSize=29,leading=36,textColor=NAVY,spaceAfter=17),
 'h1':ParagraphStyle('h1',fontName='DVB',fontSize=19,leading=24,textColor=NAVY,spaceAfter=14,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='DVB',fontSize=12,leading=16,textColor=TEAL,spaceBefore=14,spaceAfter=7,keepWithNext=True),
 'q':ParagraphStyle('q',fontName='DVB',fontSize=10.5,leading=14,textColor=NAVY,spaceAfter=5),
 'body':ParagraphStyle('body',fontName='DV',fontSize=10.3,leading=14.4,textColor=NAVY,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='DV',fontSize=8.6,leading=12,textColor=GRAY,spaceAfter=8),
 'cell':ParagraphStyle('cell',fontName='DV',fontSize=9,leading=12,textColor=NAVY),
}

def rich(value):
    escaped=html.escape(value).replace('\n','<br/>')
    escaped=re.sub(r'https?://[^\s<]+',lambda m:f'<link href="{m.group()}">{m.group()}</link>',escaped)
    return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',escaped)


def paragraph(value,style='body'):
    return Paragraph(rich(value),styles[style])


def footer(canvas,doc):
    canvas.saveState();canvas.setFillColor(GRAY);canvas.setFont('DV',8)
    canvas.drawString(18*mm,11*mm,'Jorge Palma · Defensa ML1 · Modelo rectangular v3')
    canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page));canvas.restoreState()


class GuideDoc(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name in ['h1','h2']:
            label=flowable.getPlainText();level=0 if flowable.style.name=='h1' else 1
            self._outline_counter=getattr(self,'_outline_counter',0)+1
            key=f'guide-{self._outline_counter}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label,key,level=level,closed=False)


story=[Spacer(1,18*mm),paragraph('Preparación para la defensa','title'),
    paragraph('Clasificación de artrosis de rodilla','h1'),
    paragraph('Jorge Javier Palma Martínez<br/>Machine Learning 1 (23433) · FIUNA · 2026'.replace('<br/>','\n')),
    Spacer(1,8*mm),paragraph(f'Guion de 8 a 10 minutos · 14 diapositivas<br/>{question_count} preguntas técnicas con respuestas'.replace('<br/>','\n'),'h2'),
    paragraph('Basado en el modelo guardado, el código y los resultados vigentes. Incluye matemática, decisiones técnicas, limitaciones y ensayo oral.'),
    Spacer(1,8*mm),paragraph('Modelo: SVM RBF C=1 · 288 características.'),
    paragraph('F1 macro test: 0,7155 · Accuracy: 71,31 %.'),
    paragraph('El test ya fue evaluado en una versión anterior: la reevaluación es descriptiva, sin confirmación independiente.'),
    Spacer(1,10*mm),paragraph('Cómo usarlo: primero estudiar la ficha; después ensayar el guion con cronómetro; por último practicar preguntas al azar. Las preguntas reales pueden ser diferentes.','small'),
    paragraph('Versión de estudio: '+metrics['protocol'],'small')]
lines=text.splitlines();i=0
while i<len(lines) and not lines[i].startswith('## 1.'):i+=1
bank=False;first_category=True
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('## '):
        story.extend([PageBreak(),paragraph(line[3:],'h1')])
        bank=line.startswith('## 3.');first_category=True;i+=1;continue
    if line.startswith('### '):
        if bank and not first_category:story.append(Spacer(1,4*mm))
        story.append(paragraph(line[4:],'h2'));first_category=False;i+=1;continue
    if line.startswith('!['):
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if not match:raise ValueError('Imagen Markdown inválida: '+line)
        caption,relative=match.groups()
        image_path=(source.parent/relative).resolve()
        if not image_path.is_relative_to(R.resolve()):raise ValueError('Imagen fuera del proyecto')
        image=ReportImage(str(image_path))
        ratio=min(174*mm/image.imageWidth,170*mm/image.imageHeight)
        image.drawWidth=image.imageWidth*ratio;image.drawHeight=image.imageHeight*ratio
        group=[image,Spacer(1,2*mm),paragraph(caption,'small')]
        # An outer KeepTogether is needed: a heading followed by a nested
        # KeepTogether can otherwise remain orphaned on the previous page.
        if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='h2':
            group.insert(0,story.pop())
        story.append(KeepTogether(group))
        i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[: -]+',v or '-') for v in row):rows.append(row)
            i+=1
        cells=[[paragraph(v,'cell') for v in row] for row in rows]
        table=Table(cells,colWidths=[51*mm,123*mm],repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF2F5')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#D3DEE7')),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([table,Spacer(1,4*mm)]);continue
    if re.match(r'^\*\*\d+\.',line):
        title=line.strip('*');i+=1;answer=[]
        while i<len(lines) and not lines[i].strip():i+=1
        while i<len(lines) and lines[i].strip():answer.append(lines[i].strip());i+=1
        story.append(KeepTogether([paragraph(title,'q'),paragraph(' '.join(answer))]));continue
    block=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
        if re.match(r'^\d+\. ',lines[i]):break
        block.append(lines[i].strip());i+=1
    story.append(paragraph(' '.join(block),'small' if line.startswith('T') and 'https://' in line else 'body'))

output=R/'presentation/preparacion_defensa_completa.pdf'
doc=GuideDoc(str(output),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,
    topMargin=18*mm,bottomMargin=20*mm,title=f'Defensa ML1: guion y {question_count} preguntas técnicas',author='Jorge Javier Palma Martínez')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
fingerprint=dict(protocol=metrics['protocol'],model_sha256=hashlib.sha256((R/'models/best_model.joblib').read_bytes()).hexdigest(),
    guide_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),questions=question_count,slides=14,
    svd_analysis_sha256=hashlib.sha256((R/'reports/svd_analysis.json').read_bytes()).hexdigest())
(R/'presentation/defensa_trazabilidad.json').write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
print('Guía generada:',output)
