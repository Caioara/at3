# Ouvidoria Inteligente

Protótipo de triagem semântica de manifestações cidadãs, elaborado a partir do desafio fornecido.

## Executar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app_ouvidoria.py
```

Na primeira execução com um modelo selecionado, o `sentence-transformers` pode baixar os pesos. Sem acesso ao modelo, o núcleo usa uma representação local de fallback para que a interface não pare — os resultados semânticos devem ser avaliados usando o modelo baixado.

## Entregáveis

- `manifestacoes.json`: corpus anônimo de 40 registros, incluindo cinco textos longos e pares semanticamente redundantes;
- `análise_comparativa.ipynb`, `deteccao_duplicatas.ipynb` e `chunking_manifestacoes.ipynb`: análises reproduzíveis;
- `app_ouvidoria.py`: interface com busca, base, projeção e chunking;
- `RELATORIO.md`: texto final pronto para exportação em PDF.

O limiar de duplicata começa em 0,85. Ele deve ser validado no notebook contra a anotação humana, pois similaridade não é por si só uma decisão administrativa.
