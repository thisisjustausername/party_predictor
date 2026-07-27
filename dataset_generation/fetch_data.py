# TODO: remove words containing ...

import json
import os

import requests
import xmltodict
from aiohttp.web import Response
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get('API_KEY')


def fetch_url(ressource: str, ident: str | None = None) -> Response | None:
    url = f'https://search.dip.bundestag.de/api/v1/{ressource}' + (f'/{ident}' if ident else '')
    headers = {
        'Authorization': f'ApiKey {API_KEY}',
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def find_protocols(res: Response) -> list:
    xml_docs = []
    for i in res['documents']:
        if i['dokumentart'] != 'Plenarprotokoll' or 'xml_url' not in i['fundstelle']:
            continue
        xml_docs.append(i['fundstelle']['xml_url'])

    sps = []
    for xml_doc in xml_docs:
        result = requests.get(xml_doc)
        xml_data = xmltodict.parse(result.text)
        speaks = clean_protocol(xml_data)
        speeches = speech_extraction_aggressive(speaks)
        sps.extend(speeches)
    return sps

def clean_protocol(protocol: dict) -> list:
    '''
    Cleans a Bundestag plenary protocol

    Args:
        protocol (str): text of protocol to clean
    Returns:
        list: list of unprocessed speeches
    '''
    speaks = protocol['dbtplenarprotokoll']['sitzungsverlauf']['tagesordnungspunkt']
    speaks = [i['rede'] for i in speaks if 'rede' in i]
    return speaks


def speech_extraction_aggressive(speaks: list) -> list[dict[str, dict[str, str | None]] | str]:
    '''
    Extracts speeches from a Bundestag plenary protocol

    Args:
        speaks (dict): cleaned protocol

    Returns:
        list[dict[str, dict[str, str | None]] | str]: list of speeches with talker information and speech text
    '''
    speeches = []
    for topic in speaks:
        for speech in topic:
            sp = ''
            if not isinstance(speech, dict):
                continue
            speech = speech['p']
            talker = speech[0]['redner']['name']
            talker = {'surname': talker['nachname'], 'first_name': talker['vorname'], 'party': talker.get('fraktion', None)}
            speech = speech[1:]
            for index, abstract in enumerate(speech):
                if abstract['@klasse'] == 'J_1' and (index == len(speech) - 1):
                    # print(abstract)
                    continue
                    # NOTE: also removing valid parts of speeches, but most of them are from Bundestagspräsident*in
                if '#text' in abstract:
                    sp += (('\n' if (a := abstract['@klasse']) in ['J', 'J_1'] else ' ' if a == 'O' else ' ') + abstract['#text'])
            speeches.append({'talker': talker, 'speech': sp.strip()})

    return speeches # type: ignore


if __name__ == '__main__':
    ressource = 'plenarprotokoll'
    # ident = '325539'
    ident = None
    res = fetch_url(ressource, ident)
    result = find_protocols(res)
    print(len(result))

    with open('datasets/protocols_speeches.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
