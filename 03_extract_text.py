import json
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------
# Config
# ---------------------------
URLS_PATH = "data/urls_filtradas.txt"
OUT_PATH = "data/pages.jsonl"

MAX_PAGES = 80      # para empezar (sube después)
SLEEP_S = 0.2       # para no saturar tu servidor
MIN_TEXT_CHARS = 300

@dataclass
class PageDoc:
    url: str
    title: str
    text: str

# ---------------------------
# Helpers
# ---------------------------
def fetch_html(url: str) -> str:
    """
    Descarga HTML de una URL.
    """
    r = requests.get(url, timeout=25, headers={"User-Agent": "DatakyBot/1.0"})
    r.raise_for_status()
    return r.text


def normalize_whitespace(text: str) -> str:
    """
    Convierte muchos espacios/saltos en uno solo.
    """
    return re.sub(r"\s+", " ", text).strip()

def read_urls(path: str) -> list[str]:
    """
    Lee URLs desde un txt (una por línea), ignora vacías.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    

def remove_noise(soup: BeautifulSoup) -> None:
    """
    Elimina elementos que NO aportan al contenido:
    scripts, estilos, svg, etc.
    También puedes agregar aquí banners, nav, footer, etc.
    """
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()


def pick_main_container(soup: BeautifulSoup):
    """
    Intenta seleccionar el contenedor que suele traer el contenido principal.
    Orden de prioridad:
    - <main> o <article>
    - un div típico de contenido (algunos temas lo usan)
    - <body> como fallback
    """
    return (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"class": re.compile(r"(content|post|entry|page)", re.I)})
        or soup.body
    )


def extract_title(soup: BeautifulSoup) -> str:
    """
    Título preferido: <title> del documento.
    """
    if soup.title:
        return soup.title.get_text(strip=True)
    return ""

def extract_clean_text(html: str) -> tuple[str, str]:
    """
    Convierte HTML a texto "útil".
    Devuelve (title, text).
    """
    soup = BeautifulSoup(html, "html.parser")
    remove_noise(soup)

    title = extract_title(soup)

    container = pick_main_container(soup)
    if container:
        text = container.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    text = normalize_whitespace(text)
    return title, text


def save_jsonl(docs: Iterable[PageDoc], out_path: str) -> None:
    """
    Guarda documentos en formato JSONL:
    1 línea = 1 JSON
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")



# ---------------------------
# Main
# ---------------------------
def main():
    urls = read_urls(URLS_PATH)[:MAX_PAGES]
    print(f"📄 URLs a procesar: {len(urls)} (MAX_PAGES={MAX_PAGES})")

    docs: list[PageDoc] = []

    for i, url in enumerate(urls, start=1):
        try:
            html = fetch_html(url)
            title, text = extract_clean_text(html)

            # filtro: descarta páginas con muy poco texto (a veces son “thin pages”)
            if len(text) < MIN_TEXT_CHARS:
                print(f"⚠️ {i}/{len(urls)} texto muy corto, se omite: {url}")
                continue

            docs.append(PageDoc(url=url, title=title, text=text))
            print(f"✅ {i}/{len(urls)} OK: {url} ({len(text)} chars)")
            time.sleep(SLEEP_S)

        except Exception as e:
            print(f"❌ {i}/{len(urls)} ERROR: {url} -> {e}")

    save_jsonl(docs, OUT_PATH)
    print(f"\n🎉 Listo. Guardados {len(docs)} docs en {OUT_PATH}")


if __name__ == "__main__":
    main()