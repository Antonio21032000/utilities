elif t == "CPLE3":
    price_cple3 = prices.get("CPLE3", np.nan)
    price_cple5 = prices.get("CPLE5", np.nan)
    shares_cple3 = shares_classes["CPLE3"]
    shares_cple5 = shares_classes["CPLE5"]

    # preço exibido na tabela/grafico
    price = price_cple3

    # nº de ações exibidas = soma das duas classes
    shares = shares_cple3 + shares_cple5

    # market cap econômico = soma dos 2 blocos
    parts = []
    if pd.notna(price_cple3):
        parts.append(price_cple3 * shares_cple3)
    if pd.notna(price_cple5):
        parts.append(price_cple5 * shares_cple5)

    mc = sum(parts) if parts else np.nan

























