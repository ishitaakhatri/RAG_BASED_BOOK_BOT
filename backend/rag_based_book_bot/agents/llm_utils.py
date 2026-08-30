"""
Utility to extract text from LLM responses that may contain thinking parts.
Gemma 4 models return response.content as a list of parts (thinking + text)
instead of a plain string. This helper normalizes the response.
"""

def extract_text_from_response(content):
    """
    Extract the text content from an LLM response.
    Handles both plain string responses and Gemma 4's thinking-mode
    list-of-parts responses.
    
    Args:
        content: Either a string or a list of dicts with 'type' and 'text'/'thinking' keys
    
    Returns:
        str: The extracted text content
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # Gemma 4 returns [{'type': 'thinking', 'thinking': '...'}, {'type': 'text', 'text': '...'}]
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text' and 'text' in part:
                    text_parts.append(part['text'])
                elif 'text' in part:
                    text_parts.append(part['text'])
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts) if text_parts else str(content)
    
    # Fallback: convert to string
    return str(content)
