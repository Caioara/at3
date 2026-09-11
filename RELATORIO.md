# Relatório — Ouvidoria Inteligente

## Decisões

O protótipo usa Sentence Transformers multilíngue como representação principal, com `paraphrase-multilingual-MiniLM-L12-v2` como padrão e BGE-PT como alternativa. A busca é por cosseno e retorna cinco resultados. Para execução sem download de modelo, há fallback determinístico por *hashing* de n-gramas; ele mantém o app funcional, mas não substitui embeddings semânticos.

## Representações e duplicatas

BoW distingue apenas a coocorrência literal e gera vetores grandes. TF-IDF reduz o peso de palavras frequentes, mas ainda deixa paráfrases como “cratera” e “buraco” distantes. Embeddings aproximam essas formulações e por isso são a escolha para triagem. O limiar inicial de 0,85 privilegia precisão; deve ser calibrado com uma amostra rotulada, observando pares no percentil 90 e os falsos negativos.

## Chunking

Os cinco textos longos são separados recursivamente, primeiro em parágrafos e sentenças. Overlap mantém a transição entre trechos, evitando que uma evidência e sua localização fiquem separadas. A configuração 300/50 é adequada para denúncias curtas; 450/80 preserva melhor narrativas com contexto e providência solicitada. Chunks da mesma manifestação devem surgir próximos no PCA/t-SNE, embora isso seja uma indicação visual, não uma métrica de qualidade.

### Entrega 3 — respostas solicitadas

**Como o overlap influencia a coesão semântica?** O overlap repete o final de um chunk no começo do seguinte. Assim, referências como “a situação”, o local da denúncia ou sua consequência continuam disponíveis nos dois vetores. Sem overlap, uma frase pode terminar em um chunk e a evidência que dá sentido a ela pode ficar isolada no próximo; isso enfraquece a recuperação semântica.

**Qual configuração preserva melhor o sentido?** Foram testadas as configurações `(chunk_size=300, chunk_overlap=50)` e `(chunk_size=450, chunk_overlap=80)`. A segunda é a escolhida para as denúncias longas: os 450 caracteres normalmente acomodam o problema, seus impactos e o pedido do cidadão no mesmo trecho. Por exemplo, em M033 a descrição da cratera, o risco aos veículos e o pedido de tapa-buraco podem permanecer juntos. A opção 300/50 oferece buscas mais granulares, mas pode separar a causa (“a cratera não é visível”) da consequência (“veículos entram na faixa contrária”).

**Visualização 2D e proximidade.** O notebook `chunking_manifestacoes.ipynb` gera uma projeção PCA dos chunks da configuração 450/80, colorindo cada manifestação. A leitura esperada é que chunks da mesma manifestação formem grupos próximos, pois compartilham local, problema e vocabulário. Se um chunk se afastar, ele pode conter uma providência ou detalhe específico; portanto, o gráfico é evidência exploratória e deve ser conferido pela leitura dos textos. Execute a última célula do notebook para incluir o gráfico produzido no arquivo final.

## Dificuldades e aprendizados

Categorias administrativas não são temas semanticamente puros: iluminação, por exemplo, pode ser infraestrutura ou segurança. A visualização deve portanto orientar a triagem, e não substituir revisão humana. O projeto prioriza cache no Streamlit, tratamento de indisponibilidade de modelo e transparência sobre limites do limiar.

> Para a entrega final, exporte este arquivo para PDF pelo editor de texto, mantendo-o em até cinco páginas.
