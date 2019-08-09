from bs4 import BeautifulSoup


def remove_quotes(document, get_header=False):
    if get_header: return 'quotes'
    soup = BeautifulSoup(document, 'html.parser')

    quotes = soup.find_all('div', {'style': 'margin:20px; margin-top:5px; '})
    quote_citation = soup.find_all('div', {'align': 'right'})

    count = len(quotes)
    for quote in quotes:
        quote.decompose()
    for citation in quote_citation:
        citation.decompose()

    return count, soup.text
