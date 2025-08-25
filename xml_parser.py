import xml.etree.ElementTree as ET
import json
import argparse
import re
import os

def get_full_text(element):
    """Recursively gets all text from an element and its children."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()

def parse_xml_to_json(xml_path):
    """
    Parses a TEI XML file and extracts its content into a structured JSON format.
    """
    # Namespace for TEI XML
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        return {"error": f"Failed to parse XML: {e}"}

    # doc_id from filename
    doc_id = os.path.splitext(os.path.basename(xml_path))[0]
    if doc_id.endswith('.tei'):
        doc_id = doc_id[:-4]

    # Title
    title_element = root.find('.//tei:titleStmt/tei:title[@type="main"]', ns)
    title = get_full_text(title_element)

    # Abstract
    abstract_element = root.find('.//tei:profileDesc/tei:abstract//tei:p', ns)
    abstract = get_full_text(abstract_element)

    # Sections
    sections = []
    body = root.find('.//tei:body', ns)
    if body is not None:
        for div in body.findall('tei:div', ns):
            head_element = div.find('tei:head', ns)
            if head_element is not None:
                heading = get_full_text(head_element)
                
                # Some headings have 'n' attribute for section number
                if 'n' in head_element.attrib:
                    # Prepend section number if it exists
                    heading = head_element.attrib['n'] + " " + heading
                
                paragraphs = [get_full_text(p) for p in div.findall('.//tei:p', ns)]
                paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs if p] # clean up and remove empty paragraphs

                if heading and paragraphs:
                    sections.append({
                        "heading": heading.strip(),
                        "paragraphs": paragraphs
                    })

    # Figures
    figures = []
    # Find all figure tags anywhere in the document
    for fig in root.findall('.//tei:figure', ns):
        fig_id_raw = fig.get('{http://www.w3.org/XML/1998/namespace}id')
        if not fig_id_raw:
            continue
        
        # Reformat id, e.g., fig_0 -> Fig0
        fig_id = fig_id_raw.replace('_', ' ').title().replace(' ', '')

        caption_element = fig.find('tei:figDesc', ns)
        if caption_element is not None:
            caption = get_full_text(caption_element)
            # Clean up caption, e.g., remove "Figure X:" prefix
            caption = re.sub(r'^(Figure|Fig)\.?\s*\d*\s*[:.]?\s*', '', caption, flags=re.IGNORECASE).strip()
            
            figures.append({
                "id": fig_id,
                "caption": caption
            })

    output = {
        "doc_id": doc_id,
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "figures": figures
    }

    return output

def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(description="Parse a TEI XML file to structured JSON.")
    parser.add_argument("xml_path", help="Path to the XML file to parse.")
    args = parser.parse_args()

    parsed_data = parse_xml_to_json(args.xml_path)
    print(json.dumps(parsed_data, indent=2))

if __name__ == "__main__":
    main()
