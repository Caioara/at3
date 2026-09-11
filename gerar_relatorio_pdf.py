"""Gera o PDF final da entrega."""
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import hashlib
import reportlab.pdfbase.pdfdoc as pdfdoc

# Compatibilidade do ReportLab 4.4 com OpenSSL/Python 3.8 no Windows.
pdfdoc.md5 = lambda usedforsecurity=False: hashlib.md5()

doc = SimpleDocTemplate("RELATORIO.pdf", pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=1.7*cm, bottomMargin=1.7*cm)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TituloProjeto", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=HexColor("#123342"), spaceAfter=14))
styles.add(ParagraphStyle(name="SecaoProjeto", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=HexColor("#0c6670"), spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="CorpoProjeto", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.2, leading=14.5, textColor=HexColor("#1c2d35"), spaceAfter=8))

conteudo = [
    ("Relatório — Ouvidoria Inteligente", "title"),
    ("Protótipo de triagem semântica de manifestações cidadãs", "body"),
    ("Decisões técnicas", "heading"),
    ("O protótipo usa Sentence Transformers multilíngue como representação principal, com paraphrase-multilingual-MiniLM-L12-v2 como padrão e BGE-PT como alternativa. A busca usa similaridade de cosseno e retorna os cinco resultados mais relacionados. Sem acesso ao modelo, há fallback local por n-gramas para manter a interface disponível; a avaliação semântica deve ser feita com o modelo carregado.", "body"),
    ("Representações e duplicatas", "heading"),
    ("BoW reconhece termos literais e TF-IDF reduz o peso de palavras frequentes, mas ambos dependem de vocabulário compartilhado. Embeddings aproximam paráfrases como ‘cratera’ e ‘buraco’. O limiar inicial de 0,85 favorece precisão, porém foi conservador para este corpus: a validação de referência revela falsos negativos, justificando uma segunda faixa de 0,30 a 0,50 para revisão humana.", "body"),
    ("Chunking de manifestações longas", "heading"),
    ("As cinco maiores manifestações (M033 a M037) possuem mais de 500 caracteres e são separadas com RecursiveCharacterTextSplitter. Foram comparadas as configurações 300/50 e 450/80. O overlap repete a fronteira entre chunks e preserva local, causa e consequência. A configuração 450/80 preserva melhor denúncias narrativas: em M033 mantém o risco da cratera e o pedido de sinalização no mesmo contexto. A projeção PCA mostra que chunks do mesmo relato tendem a ficar próximos; pontos distantes podem representar um pedido ou detalhe específico.", "body"),
    ("Aplicação e aprendizados", "heading"),
    ("O app Streamlit reúne busca semântica, consulta à base, matriz de similaridade, espaço vetorial e chunking interativo. A projeção PCA/t-SNE é exploratória: categorias administrativas nem sempre correspondem a temas semanticamente puros. Iluminação, por exemplo, pode aparecer como infraestrutura ou segurança. O sistema apoia a triagem humana, não substitui a leitura e a decisão administrativa.", "body"),
]

elementos = []
for texto, tipo in conteudo:
    estilo = styles["TituloProjeto"] if tipo == "title" else styles["SecaoProjeto"] if tipo == "heading" else styles["CorpoProjeto"]
    elementos.append(Paragraph(texto, estilo))
    if tipo == "title": elementos.append(Spacer(1, 4))
doc.build(elementos)
print("PDF gerado: RELATORIO.pdf")
