from bs4 import BeautifulSoup


def remove_quotes(document, get_header=False):
    if get_header: return 'quotes'
    soup = BeautifulSoup(document)

    quotes = soup.find_all('div', {'style': 'margin:20px; margin-top:5px; '})
    quote_citation = soup.find_all('div', {'align': 'right'})

    if len(quotes) != len(quote_citation):
        raise ValueError('Number of quotes not equal to number of quote citations')
    count = len(quotes)

    for quote, citation in zip(quotes, quote_citation):
        quote.decompose()
        citation.decompose()

    return count, str(soup)
