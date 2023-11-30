import requests
import json
import re


def fetch_article_content(article_title):
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "extracts",
        "titles": article_title,
        "explaintext": True,
        "format": "json"
    }

    url_title = article_title
    response = requests.get(url=URL, params=PARAMS)
    data = response.json()
    pages = data["query"]["pages"]

    # Since there's typically one page per title, you can get the page ID like this
    page_id = next(iter(pages))
    content = pages[page_id].get("extract", "")
    title = pages[page_id].get("title", "")
    wiki_url = f"https://en.wikipedia.org/wiki/{url_title}"
    return title, page_id, content


def tokenize_and_clean(text):
    # Tokenize by whitespace
    tokens = text.split()

    # Remove unwanted LaTeX commands or mathematical expressions from tokens
    # cleaned_tokens = [token for token in tokens if not re.match(r'\\[a-zA-Z]+', token)]
    cleaned_tokens = [token for token in tokens if not re.match(r'({.*?})|(\\[a-zA-Z]+)', token)]

    # Rejoin the cleaned tokens
    cleaned_text = ' '.join(cleaned_tokens)
    # cleaned_text = re.sub(r"({.*?})","", clean_text)
    return cleaned_text


def remove_nested_curly_braces(text):
    """
    Remove nested curly braces and their content up to arbitrary levels of nesting.
    """
    # Stack to keep track of the indices of opening braces
    stack = []
    # List to keep the ranges of indices to remove
    to_remove = []
    # Convert the text into a list of characters for easier manipulation
    text_list = list(text)

    # Iterate over the text and check each character
    for i, char in enumerate(text_list):
        if char == "\\{":
            stack.append(i)
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:  # Only add to removal list if we are back to base level
                    to_remove.append((start, i))

    # Remove the content in reverse order to not mess up the indices
    for start, end in reversed(to_remove):
        del text_list[start:end + 2]  # +1 to include the closing brace

    # Convert the list of characters back into a string
    return ''.join(text_list)


def remove_whitespace(text):
    # Replace \n and \t with an empty string
    cleaned_text = re.sub(r'[\n\t]+', '', text)
    return cleaned_text