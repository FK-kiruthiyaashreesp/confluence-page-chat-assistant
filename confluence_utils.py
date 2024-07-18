import re
from atlassian import Confluence
from bs4 import BeautifulSoup

def get_confluence_client(url, username, token):
    try:
        return Confluence(url=url, username=username, password=token)
    except Exception as e:
        raise Exception(f"Error initializing Confluence client: {e}")

def get_page_id_or_path(url_or_id):
    if url_or_id.isdigit():
        return url_or_id
    else:
        match = re.search(r'pages/(\d+)', url_or_id)
        if match:
            return match.group(1)
        else:
            return None

def get_confluence_page_content(confluence, page_id):
    try:
        content = confluence.get_page_by_id(page_id, expand='body.storage')
        return content
    except Exception as e:
        raise Exception(f"Error fetching content for Confluence page ID {page_id}: {e}")

def html_to_plain_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)
